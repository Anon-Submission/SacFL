#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from loss import DistillationLoss
from elastic_weight_consolidation import ElasticWeightConsolidation
import copy
from importlib import import_module
import pickle as pkl
from tqdm import tqdm
import math
import os

from utils import init_network, draw_model_heatmap


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # # for image
        # return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)
        # # for text
        return torch.tensor(image), torch.tensor(label)


class MyDataSet(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        # # for image
        # return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)
        # # for text
        return torch.tensor(image), torch.tensor(label)


class GlobalDataSetSplit(Dataset):
    def __init__(self, dataset, idxs, local_models, device):
        self.device = device
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.local_models = local_models
        for i in self.local_models:
            i.eval()
        self.model_idx = 0

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        temp_image, label = torch.tensor(image).to(self.device), torch.tensor(label).to(self.device)
        temp_image = temp_image.view(1, temp_image.shape[0])
        soft_label = self.local_models[self.model_idx](temp_image)
        self.model_idx = (self.model_idx + 1) % len(self.local_models)
        return torch.tensor(image), soft_label.view(soft_label.shape[1])


class LocalUpdate(object):
    def __init__(self, args, train_data, test_data, user_task, user_data, logger, current_task, lr):
        self.train_data = train_data
        self.test_data = test_data
        self.user_task = user_task
        self.user_data = user_data
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader, self.user_data_test_task = self.train_val_test(current_task)
        self.device = args.device
        self.current_task = current_task
        self.criterion = F.cross_entropy
        self.lr = lr

    def train_val_test(self, current_task):
        user_train_data = []
        user_data_test_task = []
        test_data = []
        if self.args.scenario == 'class':
            for i, j in enumerate(self.user_data[0]):
                for k in self.user_task[current_task]:
                    user_train_data.append(self.train_data[k][self.user_data[k][i]])
            random.shuffle(user_train_data)
            for j in range(current_task+1):
                user_data_test_task.extend(self.user_task[j])
            for i in user_data_test_task:
                test_data.extend(self.test_data[i])
        elif self.args.scenario == 'domain':
            for i, j in enumerate(self.user_data[0]):
                for k in self.user_task[current_task]:
                    user_train_data.append(self.train_data[current_task*10+k][self.user_data[k][i]])
            random.shuffle(user_train_data)
            for j in range(current_task+1):
                for i in range(self.args.num_classes):
                    test_data.extend(self.test_data[j*10+i])

        trainloader = DataLoader(user_train_data, batch_size=self.args.local_bs, shuffle=False, drop_last=True)
        # idxs_test -> idxs_val
        if self.args.model_name == 'TextCNN':
            batch_size = 1000
        elif self.args.model_name == 'CNN_Cifar10':
            batch_size = 1000
        elif self.args.model_name == 'CNN_Cifar100':
            batch_size = 100
        elif self.args.model_name == 'LeNet_FashionMNIST':
            batch_size = 1000
        validloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
        testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
        return trainloader, validloader, testloader, user_data_test_task

    def detect_shift(self, model, datasets, idx, global_round, change_dict, last_loss, task_shift):
        # datasets_loader = DataLoader(datasets, batch_size=16, shuffle=False)
        datasets_loader = datasets
        last_task_global_model = copy.deepcopy(model)
        current_dict = model.state_dict()
        backbone_dict = torch.load('./save_model/global_model.pth')
        pretrained_dict = torch.load('./save_model/Client_{}.pth'.format(idx))

        if task_shift == True:
            for index, (name, param) in enumerate(pretrained_dict.items()):
                if index == len(pretrained_dict)-2:
                    weight_shape = pretrained_dict[name].shape
                    current_dict[name][:weight_shape[0], :] = pretrained_dict[name]
                elif index == len(pretrained_dict)-1:
                    bias_shape = pretrained_dict[name].shape
                    current_dict[name][:bias_shape[0]] = pretrained_dict[name]
                else:
                    current_dict[name] = backbone_dict[name]
        last_task_global_model.load_state_dict(current_dict)

        loss = 0
        for batch_idx, (x, target) in enumerate(datasets_loader):
            if batch_idx <= 9:
                x, target = x.to(self.device), target.to(self.device)

                x.requires_grad = False
                target.requires_grad = False

                if 'cifar' in self.args.model_name.lower():
                    x = x.to(torch.float32)
                    sub_model = torch.nn.Sequential(*list(model.children())[:-1])
                    pro1 = sub_model(x)
                    sub_last_task_global_model = torch.nn.Sequential(*list(last_task_global_model.children())[:-1])
                    pro2 = sub_last_task_global_model(x)
                else:
                    _, pro1 = model(x)
                    _, pro2 = last_task_global_model(x)
                # diff = cos(pro1, pro2)
                pdist = nn.PairwiseDistance(p=1)
                diff = pdist(pro1, pro2)
                gap = torch.norm(diff)
                loss += gap
        # print('loss. vs. thresh:', "{:.2e}".format(loss))
        # thresh = maxval
        if 'MNIST' in self.args.model:
            thresh = 500
        elif self.args.model == 'CNN_Cifar10':
            thresh = 100
        elif self.args.model == 'CNN_Cifar100':
            thresh = 100
        elif 'Text' in self.args.model:
            thresh = 1000
        print('idx_{}_epoch_{}_thresh_{}'.format(idx,global_round,loss))
        if loss > thresh:
            shift_or_not = True
            torch.save(backbone_dict, './save_model/global_{}_history_model.pth'.format(global_round-1))
            file_path = './save_model/Epoch_{}_Client_{}_head.pth'.format(global_round - 1, idx)
            torch.save(dict(list(pretrained_dict.items())[-2:]), file_path)
            change_dict[idx].append(global_round)
        else:
            shift_or_not = False
        return loss, shift_or_not, change_dict

    def update_weights_ccfed(self, model, global_round, idx, change_dict, mu, shift_or_not, two_epoch_loss=None, task_shift=False):
        # Set mode to train model
        model.train()
        if self.current_task > 0:
            length = len(list(model.parameters()))
            for index, p in enumerate(model.modules()):
                if index < length - 2:
                    p.requires_grad_ = False
        epoch_loss = []
        epoch_acc = []
        criterion = nn.CrossEntropyLoss().cuda()

        print('shift_or_not_{}'.format(shift_or_not))

        for iter in range(self.args.local_ep):
            total = 0
            correct = 0
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                if 'cifar' in self.args.model_name.lower():
                    images = images.to(torch.float32)
                    out = model(images)
                else:
                    out, pro1 = model(images)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                loss = self.criterion(out, labels)
                loss.backward()
                optimizer.step()
                # Prediction
                _, pred_labels = torch.max(out, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{}]\tLoss: {:.6f}'.format(global_round, iter, batch_idx * len(images), len(self.trainloader.dataset), loss.item()))
                self.logger.add_scalar('loss', loss.item())

                batch_loss.append(loss.item())

            epoch_acc.append(correct/total)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            if iter == self.args.local_ep-1:
                torch.save(model.state_dict(), './save_model/Client_{}.pth'.format(idx))
                torch.save(dict(list(model.state_dict().items())[-2:]), './save_model/Client_{}_head.pth'.format(idx))
            if iter == 0:
                if global_round > 0:
                    gap, shift_or_not, change_dict = self.detect_shift(model, self.trainloader, idx, global_round, change_dict, two_epoch_loss[idx], task_shift)
                    # file_path = './save_data/Epoch_{}_Client_{}_data'.format(global_round-1, idx)
                    # if os.path.exists(file_path):
                    #     os.remove(file_path)
                    gap_ = gap.item()
                    two_epoch_loss[idx] = abs(gap_)
                else:
                    gap = torch.tensor(0)
                    gap = gap.to(self.device)
                    shift_or_not = False
        print('global_round_{}_user_id_{}_train_acc_{}'.format(global_round, idx, epoch_acc[-1]))

        return model.state_dict(), sum(epoch_loss)/len(epoch_loss), change_dict, gap, shift_or_not, two_epoch_loss

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                if 'cifar' in self.args.model_name.lower():
                    images = images.to(torch.float32)
                    log_prob = model(images)
                else:
                    log_prob, _ = model(images)
                loss = self.criterion(log_prob, labels)

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(global_round, iter, batch_idx * len(images), len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_fedprox_weights(self, mu, model, global_round):
        # Set mode to train model
        model.train()
        global_model = copy.deepcopy(model)
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate)

        model.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                if 'cifar' in self.args.model_name.lower():
                    images = images.to(torch.float32)
                    output = model(images)
                else:
                    output, _ = model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                for w, w_g in zip(model.parameters(), global_model.parameters()):
                    w.grad.data += mu * (w_g.data - w.data)
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(global_round, iter, batch_idx * len(images), len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_multihead(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        epoch_acc = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            total, correct = 0, 0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                if 'cifar' in self.args.model_name.lower():
                    images = images.to(torch.float32)
                    log_prob = model(images)
                else:
                    log_prob, _ = model(images)
                # log_prob = log_probs[self.current_task]
                loss = self.criterion(log_prob, labels)

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                _, pred_labels = torch.max(log_prob, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(correct / total)

        print('global_round_{}_train_acc_{}'.format(global_round, epoch_acc[-1]))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_CFeD_new(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        epoch = self.args.local_ep
        for iter in range(epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                if 'cifar' in self.args.model_name.lower():
                    images = images.to(torch.float32)
                    log_prob = model(images)
                else:
                    log_prob, _ = model(images)
                loss = self.criterion(log_prob, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), model

    def update_weights_CFeD_old(self, model, initial_model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        # Set optimizer for the local updates

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        dis_loss = DistillationLoss()
        old_targets = []
        model.eval()
        with torch.no_grad():
            for trains, labels in self.trainloader:
                trains, labels = trains.to(self.device), labels.to(self.device)
                if 'cifar' in self.args.model_name.lower():
                    trains = trains.to(torch.float32)
                    output = initial_model(trains)
                else:
                    output, _ = initial_model(trains)
                old_targets.append(output)

        model.train()
        for iter in range(int(self.args.local_ep)):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # labels -= int(self.args.task_list[self.current_task][0])
                model.zero_grad()
                if 'cifar' in self.args.model_name.lower():
                    images = images.to(torch.float32)
                    output = model(images)
                else:
                    output, _ = model(images)
                loss = dis_loss(output, old_targets[batch_idx], 2.0, 0.1)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), model

    def update_weights_lwf(self, model, initial_model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        epoch_acc = []
        # Set optimizer for the local updates

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        dis_loss = DistillationLoss()
        old_targets = []
        model.eval()
        if self.current_task > 0:
            with torch.no_grad():
                for trains, labels in self.trainloader:
                    trains, labels = trains.to(self.device), labels.to(self.device)
                    if 'cifar' in self.args.model_name.lower():
                        trains = trains.to(torch.float32)
                        output = initial_model(trains)
                    else:
                        output, _ = initial_model(trains)
                    old_targets.append(output)

        model.train()
        for iter in range(int(self.args.local_ep)):
            batch_loss = []
            correct, total = 0, 0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # labels -= int(self.args.task_list[self.current_task][0])
                model.zero_grad()
                if 'cifar' in self.args.model_name.lower():
                    images = images.to(torch.float32)
                    output = model(images)
                else:
                    output, _ = model(images)
                loss = self.criterion(output, labels)
                if self.current_task > 0:
                    loss += dis_loss(output, old_targets[batch_idx], 2.0, 0.1)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                _, pred_labels = torch.max(output, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
            train_acc = correct/total
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(train_acc)
        print('global_round_{}_train_acc_{}'.format(global_round, epoch_acc[-1]))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), model, train_acc

    def update_weights_ewc(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        epoch_acc = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        ewc = ElasticWeightConsolidation(model, weight=100000)
        if self.current_task > 0:
            ewc.register_ewc_params(self.args, self.trainloader, self.args.local_bs, len(self.trainloader))
            
        model.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            total, correct = 0, 0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()

                if 'cifar' in self.args.model_name.lower():
                    images = images.to(torch.float32)
                    log_probs = model(images)
                else:
                    log_probs, _ = model(images)

                loss = self.criterion(log_probs, labels)
                if self.current_task > 0:
                    loss += ewc.consolidation_loss(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                _, pred_labels = torch.max(log_probs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(correct/total)
        print('global_round_{}_train_acc_{}'.format(global_round, epoch_acc[-1]))
        
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
        
    def update_weights_DMC_new(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        epoch = self.args.local_ep
        # if self.args.server_distillation and self.current_task > 0:
        #     epoch *= 3
        for iter in range(epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                if 'cifar' in self.args.model_name.lower():
                    images = images.to(torch.float32)
                    log_prob = model(images)
                else:
                    log_prob, _ = model(images)

                loss = self.criterion(log_prob, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), model

    def update_weights_DMC_combine(self, global_model, new_local_model, global_round):
        # Set mode to train model
        # model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        # optimizer = torch.optim.Adam(global_model.parameters(), lr=self.args.learning_rate * 10)
        # dmc_loss = DMCLoss(self.current_task)
        dmc_loss = nn.MSELoss()
        old_targets, new_targets = [], []
        global_model.eval()
        new_local_model.eval()
        with torch.no_grad():
            for trains, labels in self.trainloader:
                trains, labels = trains.to(self.device), labels.to(self.device)
                if 'cifar' in self.args.model_name.lower():
                    trains = trains.to(torch.float32)
                    out1 = global_model(trains)
                    out2 = new_local_model(trains)
                else:
                    out1, _ = global_model(trains)
                    out2, _ = new_local_model(trains)
                old_targets.append(out1)
                new_targets.append(out2)

        new_global_model = copy.deepcopy(global_model)
        optimizer = torch.optim.Adam(new_global_model.parameters(), lr=self.lr)
        new_global_model.train()
        
        init_network(new_global_model)
        for iter in range(self.args.local_ep * 2):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                new_global_model.zero_grad()
                if 'cifar' in self.args.model_name.lower():
                    images = images.to(torch.float32)
                    log_prob = new_global_model(images)
                else:
                    log_prob, _ = new_global_model(images)

                outputs_old = old_targets[batch_idx]
                outputs_cur = new_targets[batch_idx]
                outputs_old -= outputs_old.mean(dim=1).reshape(images.shape[0], -1)
                outputs_cur -= outputs_cur.mean(dim=1).reshape(images.shape[0], -1)
                # outputs_tot = torch.cat((outputs_old, outputs_cur), dim=1)
                outputs_tot = outputs_old + outputs_cur
                loss = dmc_loss(log_prob, outputs_tot)

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return new_global_model.state_dict(), sum(epoch_loss) / len(epoch_loss), new_global_model

    def inference(self, model, idx):
        x = import_module('models.' + self.args.model_name)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        his_task = []
        his_task_dict = {}
        test_model = copy.deepcopy(model)
        test_model_state_dict = model.state_dict()
        if self.args.paradigm == 'ccfed':
            head = torch.load('./save_model/Client_{}_head.pth'.format(idx))
            for index, (key, v) in enumerate(test_model_state_dict.items()):
                if index >= len(test_model_state_dict) - 2:
                    test_model_state_dict[key] = head[key]
        test_model.load_state_dict(test_model_state_dict)
        diff_task_acc = {}
        for task in range(self.current_task+1):
            diff_task_acc[task] = []

        for i in range(self.current_task+1):
            his_task.extend(self.user_task[i])
            his_task_dict[i] = self.user_task[i]
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            if self.args.paradigm == 'ccfed':
                if 'class' in self.args.scenario:
                    for k, cls in enumerate(his_task):
                        if cls in labels:
                            task_id = [key for key, value in his_task_dict.items() if cls in value]
                    #         cls_idx = self.user_task[task_id[0]].index(cls)
                    #         index = torch.where(labels == cls)
                    #         for ele in index:
                    #             labels[ele] = labels[ele] - cls + cls_idx
                else:
                    task_id = [batch_idx//10]
                if self.current_task > 0 and task_id[0] < self.current_task:
                    test_model = copy.deepcopy(model)
                    new_dict = test_model.state_dict()
                    head = torch.load('./save_model/Epoch_{}_Client_{}_head.pth'.format((task_id[0]+1)*self.args.num_epochs-1, idx))
                    for index, (key, v) in enumerate(new_dict.items()):
                        if index < len(new_dict) - 2:
                            new_dict[key] = test_model_state_dict[key]
                        if index >= len(new_dict) - 2:
                            new_dict[key] = head[key]
                    test_model.load_state_dict(new_dict)
                else:
                    test_model = copy.deepcopy(model)
            if 'cifar' in self.args.model_name.lower():
                images = images.to(torch.float32)
                log_prob = test_model(images)
            else:
                log_prob, _ = test_model(images)
            if isinstance(log_prob, list):
                log_prob = log_prob[self.current_task]
                labels -= int(self.args.task_list[self.current_task][0])
            batch_loss = self.criterion(log_prob, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(log_prob, 1)
            pred_labels = pred_labels.view(-1)

            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            if self.args.paradigm == 'ccfed':
                diff_task_acc[task_id[0]].append(torch.sum(torch.eq(pred_labels, labels)).item()/len(labels))
                # print('currenttask_{}_userid_{}_testtaskid_{}_label_{}_correctnum_{}'.format(self.current_task, idx,
                #                                                                              task_id[0], labels[0],
                #                                                                              torch.sum(
                #                                                                                  torch.eq(pred_labels,
                #                                                                                           labels)).item()))
        # if self.args.paradigm == 'ccfed':
        #     diff_task_acc_ls = []
        #     for k, v in diff_task_acc.items():
        #         diff_task_acc_ls.append(sum(v)/len(v))
        #     diff_task_acc_ls_1 = [val/sum(diff_task_acc_ls) for val in diff_task_acc_ls]
        #     diff_task_acc_ls_1.reverse()
            
        accuracy = correct/total
        # print(accuracy)
        # if self.args.paradigm == 'ccfed':
        #     return accuracy, loss, diff_task_acc_ls_1
        # else:
        #     return accuracy, loss
        return accuracy, loss


class GlobalUpdate(object):
    def __init__(self, args, train_data, idxs, logger, current_task, local_models):
        self.args = args
        self.logger = logger
        self.local_models = local_models
        self.device = args.device
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            train_data, list(idxs))
        self.current_task = current_task
        self.criterion = F.cross_entropy

    def train_val_test(self, dataset, idxs):
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):]

        trainloader = DataLoader(GlobalDataSetSplit(dataset, idxs_train, local_models=self.local_models, device=self.device),
                                 batch_size=128, shuffle=False)
        validloader = DataLoader(GlobalDataSetSplit(dataset, idxs_val, local_models=self.local_models, device=self.device),
                                 batch_size=int(len(idxs_val) / 10), shuffle=False)
        # idxs_test -> idxs_val
        testloader = DataLoader(GlobalDataSetSplit(dataset, idxs_val, local_models=self.local_models, device=self.device),
                                batch_size=int(len(idxs_val) / 10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        dis_loss = DistillationLoss()
        for local_model in self.local_models:
            local_model.eval()
        epoch = self.args.local_ep
        for iter in range(int(epoch)):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                if 'cifar' in self.args.model_name.lower():
                    images = images.to(torch.float32)
                    outputs = model(images)
                else:
                    outputs, _ = model(images)

                loss = dis_loss(outputs, labels, 2.0, 0.1)

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def test_inference(args, model, test_dataset, current_task):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = F.cross_entropy
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        # if 'class' in args.scenario and args.paradigm == 'ccfed':
        #     for k, val in enumerate(args.subtract_val):
        #         if val in labels:
        #             index = torch.where(labels == val)
        #             for ele in index:
        #                 labels[ele] = labels[ele] - val + k

        # Inference
        if 'cifar' in args.model_name.lower():
            images = images.to(torch.float32)
            outputs = model(images)
        else:
            outputs, _ = model(images)

        if isinstance(outputs, list):
            outputs = outputs[current_task]
            labels -= int(args.task_list[current_task][0])
        batch_loss = criterion(outputs, labels)

        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
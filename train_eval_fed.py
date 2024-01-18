#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import build_dataset, build_iterator, get_time_dif, build_dataset_from_csv_fed, init_network
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils import build_usergroup, build_usergroup_non_iid
from update import LocalUpdate, test_inference, GlobalUpdate
from utils import average_weights
import pandas as pd
from importlib import import_module


def train_ccfed(config, model, train_dataset, dev_datasets, mu):
    x = import_module('models.' + config.model_name)
    start_time = time.time()
    logger = SummaryWriter('../logs')

    # Training
    train_loss, train_accuracy = [], []
    print_every = 1
    change_dict = {}
    for i in range(config.num_users):
        change_dict[i] = []
    train_acc_info_list = []
    diff_epoch_list = []
    shift_or_not_dict = {}
    two_epoch_loss = {}

    for i in range(config.num_users):
        two_epoch_loss[i] = 0
        shift_or_not_dict[i] = False
    current_task = -1
    for epoch in range(config.num_epochs * config.task_number):
        print('epoch_{}_two_epoch_loss_{}'.format(epoch, two_epoch_loss))
        current_task_copy = copy.deepcopy(current_task)
        current_task = epoch // config.num_epochs
        if current_task != current_task_copy:
            lr = config.learning_rate
            task_shift = True
            if current_task != 0:
                pretrained_dict = current_dict
                global_dict = model.state_dict()
                for index, (name, param) in enumerate(pretrained_dict.items()):
                    if index < len(pretrained_dict) - 2:
                        global_dict[name] = pretrained_dict[name]
                model.load_state_dict(global_dict)
        else:
            model.load_state_dict(torch.load('./save_model/global_model.pth', map_location='cuda'))
            task_shift = False
            # if epoch % 5 == 0 and epoch > 0:
            #     lr = lr * 0.9

        local_weights, local_losses, local_models = [], [], []

        print('\n | Global Training Round : {}|\n'.format(epoch))

        m = max(int(config.frac * config.num_users), 1)
        idxs_users = np.random.choice(range(config.num_users), m, replace=False)

        diff_user_list = []
        for idx in idxs_users:
            subtract_val = config.user_task[idx][current_task]
            config.subtract_val = subtract_val
            user_task = config.user_task[idx]
            user_data = config.user_data[idx]
            local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets, user_task=user_task, user_data=user_data, logger=logger, current_task=current_task, lr=lr)
            w, loss, change_dict, diff, shift_or_not, two_epoch_loss = local_model.update_weights_ccfed(model=copy.deepcopy(model), global_round=epoch, idx=idx, change_dict=change_dict, mu=mu,
                                                                                                        shift_or_not=shift_or_not_dict[idx], two_epoch_loss=two_epoch_loss, task_shift=task_shift)
            diff = diff.cpu()
            diff = diff.detach().numpy()
            if epoch > 0:
                diff_user_list.append(diff)
            shift_or_not_dict[idx] = shift_or_not
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        diff_user_avg = np.mean(diff_user_list)
        diff_epoch_list.append(diff_user_avg)

        # update global weights
        global_weights = average_weights(local_weights)
        current_dict = global_weights

        if current_task > 0:
            global_weights_ = copy.deepcopy(global_weights)
            pre_weights_ls = []
            keys_to_remove = list(global_weights_.keys())[-2:]
            for key in keys_to_remove:
                global_weights_.pop(key)
            pre_weights_ls.append(global_weights_)
            for i in range(current_task):
                pretrained_dict = torch.load('./save_model/global_{}_history_model.pth'.format((i+1)*config.num_epochs-1), map_location='cuda')
                keys_to_remove = list(pretrained_dict.keys())[-2:]
                for key in keys_to_remove:
                    pretrained_dict.pop(key)
                pre_weights_ls.append(pretrained_dict)
            avg_dict = average_weights(pre_weights_ls)

            current_dict = global_weights
            for index, (name, param) in enumerate(avg_dict.items()):
                if index < len(avg_dict) - 2:
                    current_dict[name] = copy.deepcopy(avg_dict[name])
        else:
            current_dict = global_weights

        # update global weights
        model.load_state_dict(current_dict)
        torch.save(current_dict, './save_model/global_model.pth')

        # return
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []

        model.eval()
        for c in range(config.num_users):
            acc_info = []
            subtract_val = config.user_task[c][current_task]
            config.subtract_val = subtract_val
            user_task = config.user_task[c]
            user_data = config.user_data[c]
            local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets, user_task=user_task, user_data=user_data, logger=logger, current_task=current_task, lr=lr)
            acc, loss = local_model.inference(model=model, idx=c)

            list_acc.append(acc)
            list_loss.append(loss)
            acc_info.append(current_task)
            acc_info.append(epoch)
            acc_info.append(c)
            acc_info.append(acc)
            train_acc_info_list.append(acc_info)

        train_accuracy.append(sum(list_acc) / len(list_acc)) # avg_acc of clients

        if (epoch + 1) % print_every == 0:
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch + 1))
            print('Current Task {} Training Loss : {}'.format(current_task, np.mean(np.array(train_loss))))
            print('Current Task {} Train Accuracy: {:.2f}% \n'.format(current_task, 100 * train_accuracy[-1]))

    df1 = pd.DataFrame(train_acc_info_list, columns=['Current_task', 'Epoch', 'Client_id', 'Train_acc'])
    df2 = pd.DataFrame(diff_epoch_list, columns=['Diff'])
    df1.to_csv('Results/train_acc_ccfed_tasknum_{}_{}.csv'.format(config.task_number, config.model_name))
    df2.to_csv('Results/diff_ccfed_tasknum_{}_{}.csv'.format(config.task_number, config.model_name))
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

def train(config, global_model, train_dataset, dev_datasets):
    start_time = time.time()
    logger = SummaryWriter('../logs')

    # Training
    train_loss, train_accuracy = [], []
    print_every = 1
    train_acc_info_list = []
    test_acc_info_list = []
    current_task = -1
    for epoch in tqdm(range(config.num_epochs * config.task_number)):
        current_task_copy = copy.deepcopy(current_task)
        current_task = epoch // config.num_epochs
        if current_task != current_task_copy:
            lr = config.learning_rate
        else:
            if epoch % 10 == 0 and epoch > 0:
                lr = lr * 0.9

        local_weights, local_losses = [], []
        print('\n | Global Training Round : {}|\n'.format(epoch+1))

        global_model.train()
        m = max(int(config.frac * config.num_users), 1)
        idxs_users = np.random.choice(range(config.num_users), m, replace=False)

        for idx in idxs_users:
            user_task = config.user_task[idx]
            user_data = config.user_data[idx]
            local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                      user_task=user_task, user_data=user_data, logger=logger,
                                      current_task=current_task, lr=lr)
            if config.paradigm.lower() == 'fedavg':
                w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            elif config.paradigm.lower() == 'fedprox':
                w, loss = local_model.update_fedprox_weights(0.3, model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(config.num_users):
            acc_info = []
            user_task = config.user_task[c]
            user_data = config.user_data[c]
            local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                      user_task=user_task, user_data=user_data, logger=logger,
                                      current_task=current_task, lr=lr)
            acc, loss = local_model.inference(model=global_model, idx=c)#验证集准确率

            list_acc.append(acc)
            list_loss.append(loss)
            acc_info.append(current_task)
            acc_info.append(epoch)
            acc_info.append(c)
            acc_info.append(acc)
            train_acc_info_list.append(acc_info)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
            print('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        for idx in range(config.num_users):
            for i in range(config.task_number):
                if i <= current_task:
                    config.num_classes = 10#config.task_class_length[i]
                    test_acc_info = []
                    user_task = config.user_task[idx]
                    user_data = config.user_data[idx]
                    local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets, user_task=user_task, user_data=user_data, logger=logger,
                                              current_task=i, lr=lr)
                    test_acc, test_loss = local_model.inference(model=global_model, idx=idx)
                    print("Current Task {} Test Task {} Accuracy: {:.2f}%".format(current_task, i, 100 * test_acc))
                    # writer.add_scalar("acc/dev{}".format(i), 100 * test_acc, epoch)
                    test_acc_info.append(current_task)
                    test_acc_info.append(epoch)
                    test_acc_info.append(i)
                    test_acc_info.append(test_acc)
                    test_acc_info_list.append(test_acc_info)

    df1 = pd.DataFrame(train_acc_info_list, columns=['Current_task', 'Epoch', 'Client_id', 'Train_acc'])
    df2 = pd.DataFrame(test_acc_info_list, columns=['Current_task', 'Epoch', 'Test_task', 'Test_acc'])
    df1.to_csv('Results/train_acc_{}_tasknum_{}_{}.csv'.format(config.paradigm, config.task_number, config.model_name))
    df2.to_csv('Results/test_acc_{}_tasknum_{}_{}.csv'.format(config.paradigm, config.task_number, config.model_name))
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

def train_lwf(config, global_model, train_dataset, dev_datasets, combin=False):
    start_time = time.time()
    logger = SummaryWriter('../logs')

    device = config.device

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # Training
    train_loss, train_accuracy = [], []
    print_every = 1

    train_acc_info_list = []
    test_acc_info_list = []
    current_task = -1
    for epoch in tqdm(range(config.num_epochs * config.task_number)):
        if epoch % config.num_epochs == 0:
            initial_model = copy.deepcopy(global_model)
        current_task_copy = copy.deepcopy(current_task)
        current_task = epoch // config.num_epochs
        if current_task != current_task_copy:
            lr = config.learning_rate
        else:
            if epoch % 10 == 0 and epoch > 0:
                lr = lr * 0.9

        local_weights, local_losses = [], []
        print('\n | Global Training Round : {}|\n'.format(epoch+1))

        global_model.train()
        m = max(int(config.frac * config.num_users), 1)
        idxs_users = np.random.choice(range(config.num_users), m, replace=False)

        for idx in idxs_users:
            user_task = config.user_task[idx]
            user_data = config.user_data[idx]
            local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                      user_task=user_task, user_data=user_data, logger=logger,
                                      current_task=current_task, lr=lr)

            w, loss, _model, train_acc = local_model.update_weights_lwf(model=copy.deepcopy(global_model), initial_model=copy.deepcopy(initial_model), global_round=epoch)
            print('Train_acc_Epoch_{}_Client_{}'.format(epoch, idx), train_acc)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(config.num_users):
            acc_info = []
            user_task = config.user_task[c]
            user_data = config.user_data[c]
            local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                      user_task=user_task, user_data=user_data, logger=logger,
                                      current_task=current_task, lr=lr)
            acc, loss = local_model.inference(model=global_model, idx=c)

            list_acc.append(acc)
            list_loss.append(loss)
            acc_info.append(current_task)
            acc_info.append(epoch)
            acc_info.append(c)
            acc_info.append(acc)
            train_acc_info_list.append(acc_info)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
            print('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        for idx in range(config.num_users):
            for i in range(config.task_number):
                if i <= current_task:
                    test_acc_info = []
                    user_task = config.user_task[idx]
                    user_data = config.user_data[idx]
                    local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                              user_task=user_task, user_data=user_data, logger=logger,
                                              current_task=i, lr=lr)
                    test_acc, test_loss = local_model.inference(model=global_model, idx=idx)
                    print("Current Task {} Test Task {} Accuracy: {:.2f}%".format(current_task, i, 100 * test_acc))
                    # writer.add_scalar("acc/dev{}".format(i), 100 * test_acc, epoch)
                    test_acc_info.append(current_task)
                    test_acc_info.append(epoch)
                    test_acc_info.append(i)
                    test_acc_info.append(test_acc)
                    test_acc_info_list.append(test_acc_info)

    df1 = pd.DataFrame(train_acc_info_list, columns=['Current_task', 'Epoch', 'Client_id', 'Train_acc'])
    df2 = pd.DataFrame(test_acc_info_list, columns=['Current_task', 'Epoch', 'Test_task', 'Test_acc'])
    df1.to_csv('Results/train_acc_lwf_tasknum_{}_{}.csv'.format(config.task_number, config.model_name))
    df2.to_csv('Results/test_acc_lwf_tasknum_{}_{}.csv'.format(config.task_number, config.model_name))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

def train_CFeD(config, global_model, train_dataset, dev_datasets, surrogate_data):
    start_time = time.time()
    logger = SummaryWriter('../logs')

    device = config.device

    surrogate_groups = build_usergroup(surrogate_data, config)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    train_loss, train_accuracy = [], []
    print_every = 1

    # random choose a shard for server distillation
    global_idx = np.random.choice(range(config.num_users), 1, replace=False)
    init_model = copy.deepcopy(global_model)

    train_acc_info_list = []
    test_acc_info_list = []
    current_task = -1
    for epoch in tqdm(range(config.num_epochs * config.task_number)):
        current_task_copy = copy.deepcopy(current_task)
        current_task = epoch // config.num_epochs
        if current_task != current_task_copy:
            lr = config.learning_rate
        else:
            if epoch % 10 == 0 and epoch > 0:
                lr = lr * 0.9

    # for epoch in tqdm(range(config.num_epochs)):
        local_weights, local_losses, local_models = [], [], []

        print('\n | Global Training Round : {}|\n'.format(epoch+1))

        global_model.train()
        m = max(int(config.frac * config.num_users), 1)
        idxs_users = np.random.choice(range(config.num_users), m, replace=False)

        for idx in idxs_users:
            user_task = config.user_task[idx]
            user_data = config.user_data[idx]
            local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                      user_task=user_task, user_data=user_data, logger=logger,
                                      current_task=current_task, lr=lr)
            w, loss, model = local_model.update_weights_CFeD_new(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_models.append(copy.deepcopy(model))

        if current_task > 0:
            idxs_users = np.random.choice(range(config.num_users), int(m*0.4), replace=False)
            for idx in idxs_users:
                user_task = config.user_task[idx]
                user_data = config.user_data[idx]
                local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                          user_task=user_task, user_data=user_data, logger=logger,
                                          current_task=current_task, lr=lr)
                w, loss, model = local_model.update_weights_CFeD_old(model=copy.deepcopy(global_model), global_round=epoch, initial_model=init_model)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
                local_models.append(copy.deepcopy(model))

        if config.server_distillation:
            local_models.append(copy.deepcopy(global_model))
        # update global weights
        print('len:', len(local_weights))
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        if config.server_distillation:
            local_model = GlobalUpdate(args=config, train_data=surrogate_data,
                                      idxs=surrogate_groups[global_idx[0]], logger=logger,
                                      current_task=current_task, local_models=local_models)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            global_model.load_state_dict(w)

        # return
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(config.num_users):
            acc_info = []
            user_task = config.user_task[c]
            user_data = config.user_data[c]
            local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                      user_task=user_task, user_data=user_data, logger=logger,
                                      current_task=current_task, lr=lr)
            acc, loss = local_model.inference(model=global_model, idx=c)

            list_acc.append(acc)
            list_loss.append(loss)
            acc_info.append(current_task)
            acc_info.append(epoch)
            acc_info.append(c)
            acc_info.append(acc)
            train_acc_info_list.append(acc_info)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
            print('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            # writer.add_scalar("loss/train_{}".format(current_task), np.mean(np.array(train_loss)), epoch)
            # writer.add_scalar("acc/train_{}".format(current_task), 100*train_accuracy[-1], epoch)

        for idx in range(config.num_users):
            for i in range(config.task_number):
                if i <= current_task:
                    test_acc_info = []
                    user_task = config.user_task[idx]
                    user_data = config.user_data[idx]
                    local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                              user_task=user_task, user_data=user_data, logger=logger,
                                              current_task=i, lr=lr)
                    test_acc, test_loss = local_model.inference(model=global_model, idx=idx)
                    print("Current Task {} Test Task {} Accuracy: {:.2f}%".format(current_task, i, 100 * test_acc))
                    test_acc_info.append(current_task)
                    test_acc_info.append(epoch)
                    test_acc_info.append(i)
                    test_acc_info.append(test_acc)
                    test_acc_info_list.append(test_acc_info)

    df1 = pd.DataFrame(train_acc_info_list, columns=['Current_task', 'Epoch', 'Client_id', 'Train_acc'])
    df2 = pd.DataFrame(test_acc_info_list, columns=['Current_task', 'Epoch', 'Test_task', 'Test_acc'])
    df1.to_csv('Results/train_acc_cfed_tasknum_{}_{}.csv'.format(config.task_number, config.model_name))
    df2.to_csv('Results/test_acc_cfed_tasknum_{}_{}.csv'.format(config.task_number, config.model_name))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

def train_ewc(config, global_model, train_dataset, dev_datasets):
    start_time = time.time()
    logger = SummaryWriter('../logs')

    device = config.device

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    train_loss, train_accuracy = [], []
    print_every = 1
    writer = SummaryWriter(log_dir=config.log_path + '/' +
                                   time.strftime('%m-%d_%H.%M_{}_{}_C[{}]_E[{}]_B[{}]_ewc'
                                                 .format(config.model_name, config.num_epochs, config.frac,
                                                         config.local_ep, config.local_bs),
                                                 time.localtime()))
    train_acc_info_list = []
    test_acc_info_list = []
    current_task = -1
    for epoch in tqdm(range(config.num_epochs * config.task_number)):
        current_task_copy = copy.deepcopy(current_task)
        current_task = epoch // config.num_epochs
        if current_task != current_task_copy:
            lr = config.learning_rate
        else:
            if epoch % 10 == 0 and epoch > 0:
                lr = lr * 0.9
        local_weights, local_losses = [], []
        print('\n | Global Training Round : {}|\n'.format(epoch+1))

        global_model.train()
        m = max(int(config.frac * config.num_users), 1)
        idxs_users = np.random.choice(range(config.num_users), m, replace=False)

        for idx in idxs_users:
            user_task = config.user_task[idx]
            user_data = config.user_data[idx]
            local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                      user_task=user_task, user_data=user_data, logger=logger,
                                      current_task=current_task, lr=lr)
            w, loss = local_model.update_weights_ewc(
                model=copy.deepcopy(global_model), global_round=epoch)

            w_temp = copy.deepcopy(w)
            for k in w.keys():
                if '__' in k:
                    w_temp.pop(k)
            local_weights.append(w_temp)
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(config.num_users):
            acc_info = []
            user_task = config.user_task[c]
            user_data = config.user_data[c]
            local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                      user_task=user_task, user_data=user_data, logger=logger,
                                      current_task=current_task, lr=lr)

            acc, loss = local_model.inference(model=global_model,idx=c)

            list_acc.append(acc)
            list_loss.append(loss)
            acc_info.append(current_task)
            acc_info.append(epoch)
            acc_info.append(c)
            acc_info.append(acc)
            train_acc_info_list.append(acc_info)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
            print('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            writer.add_scalar("loss/train_{}".format(current_task), np.mean(np.array(train_loss)), epoch)
            writer.add_scalar("acc/train_{}".format(current_task), 100*train_accuracy[-1], epoch)
        for idx in range(config.num_users):
            for i in range(config.task_number):
                if i <= current_task:
                    test_acc_info = []
                    user_task = config.user_task[idx]
                    user_data = config.user_data[idx]
                    local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                              user_task=user_task, user_data=user_data, logger=logger,
                                              current_task=i, lr=lr)
                    test_acc, test_loss = local_model.inference(model=global_model, idx=idx)
                    print("Current Task {} Test Task {} Accuracy: {:.2f}%".format(current_task, i, 100 * test_acc))
                    test_acc_info.append(current_task)
                    test_acc_info.append(epoch)
                    test_acc_info.append(i)
                    test_acc_info.append(test_acc)
                    test_acc_info_list.append(test_acc_info)

    df1 = pd.DataFrame(train_acc_info_list, columns=['Current_task', 'Epoch', 'Client_id', 'Train_acc'])
    df2 = pd.DataFrame(test_acc_info_list, columns=['Current_task', 'Epoch', 'Test_task', 'Test_acc'])
    df1.to_csv('Results/train_acc_ewc_tasknum_{}_{}.csv'.format(config.task_number, config.model_name))
    df2.to_csv('Results/test_acc_ewc_tasknum_{}_{}.csv'.format(config.task_number, config.model_name))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

def train_multihead(config, global_model, train_dataset, dev_datasets):
    start_time = time.time()
    logger = SummaryWriter('../logs')

    device = config.device

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    train_loss, train_accuracy = [], []
    print_every = 1
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M_{}_{}_C[{}]_E[{}]_B[{}]'
                                                 .format(config.model_name, config.num_epochs, config.frac,
                                                         config.local_ep, config.local_bs),
                                                 time.localtime()))
    train_acc_info_list = []
    test_acc_info_list = []
    current_task = -1
    for epoch in tqdm(range(config.num_epochs * config.task_number)):
        current_task_copy = copy.deepcopy(current_task)
        current_task = epoch // config.num_epochs
        if current_task != current_task_copy:
            lr = config.learning_rate
        else:
            if epoch % 10 == 0 and epoch > 0:
                lr = lr * 0.9
        local_weights, local_losses = [], []
        print('\n | Global Training Round : {}|\n'.format(epoch+1))

        global_model.train()
        m = max(int(config.frac * config.num_users), 1)
        idxs_users = np.random.choice(range(config.num_users), m, replace=False)

        for idx in idxs_users:
            user_task = config.user_task[idx]
            user_data = config.user_data[idx]
            local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                      user_task=user_task, user_data=user_data, logger=logger,
                                      current_task=current_task, lr=lr)

            w, loss = local_model.update_weights_multihead(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(config.num_users):
            acc_info = []
            user_task = config.user_task[c]
            user_data = config.user_data[c]
            local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                      user_task=user_task, user_data=user_data, logger=logger,
                                      current_task=current_task, lr=lr)
            acc, loss = local_model.inference(model=global_model, idx=c)

            list_acc.append(acc)
            list_loss.append(loss)
            acc_info.append(current_task)
            acc_info.append(epoch)
            acc_info.append(c)
            acc_info.append(acc)
            train_acc_info_list.append(acc_info)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
            print('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        for idx in range(config.num_users):
            for i in range(config.task_number):
                if i <= current_task:
                    test_acc_info = []
                    user_task = config.user_task[idx]
                    user_data = config.user_data[idx]
                    local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                              user_task=user_task, user_data=user_data, logger=logger,
                                              current_task=i, lr=lr)
                    test_acc, test_loss = local_model.inference(model=global_model, idx=idx)
                    print("Current Task {} Test Task {} Accuracy: {:.2f}%".format(current_task, i, 100 * test_acc))
                    test_acc_info.append(current_task)
                    test_acc_info.append(epoch)
                    test_acc_info.append(i)
                    test_acc_info.append(test_acc)
                    test_acc_info_list.append(test_acc_info)

    df1 = pd.DataFrame(train_acc_info_list, columns=['Current_task', 'Epoch', 'Client_id', 'Train_acc'])
    df2 = pd.DataFrame(test_acc_info_list, columns=['Current_task', 'Epoch', 'Test_task', 'Test_acc'])
    df1.to_csv('Results/train_acc_multihead_tasknum_{}_{}.csv'.format(config.task_number, config.model_name))
    df2.to_csv('Results/test_acc_multihead_tasknum_{}_{}.csv'.format(config.task_number, config.model_name))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

def train_DMC(config, global_model, train_dataset, dev_datasets, surrogate_data):
    start_time = time.time()
    logger = SummaryWriter('../logs')
    device = config.device

    # load dataset and user groups
    user_groups = []
    surrogate_groups = []
    # load dataset and user groups
    for i in range(config.task_number):
        if config.iid:
            user_groups_task = build_usergroup(train_dataset[i], config)
        else:
            user_groups_task = build_usergroup_non_iid(train_dataset[i], config)
        user_groups.append(user_groups_task)
        surrogate_groups_task = build_usergroup(surrogate_data, config)
        surrogate_groups.append(surrogate_groups_task)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    
    print(global_model)

    # Training
    train_loss, train_accuracy = [], []
    print_every = 1
    writer = SummaryWriter(log_dir=config.log_path + '/' +
                                   time.strftime('%m-%d_%H.%M_{}_{}_C[{}]_E[{}]_B[{}]_client'
                                                 .format(config.model_name, config.num_epochs, config.frac,
                                                         config.local_ep, config.local_bs),
                                                 time.localtime()))
    # random choose a shard for server distillation
    global_idx = np.random.choice(range(config.num_users), 1, replace=False)
    init_model = copy.deepcopy(global_model)

    train_acc_info_list = []
    test_acc_info_list = []
    current_task = -1
    for epoch in tqdm(range(config.num_epochs * config.task_number)):
        current_task_copy = copy.deepcopy(current_task)
        current_task = epoch // config.num_epochs
        if current_task != current_task_copy:
            lr = config.learning_rate
        else:
            if epoch % 10 == 0 and epoch > 0:
                lr = lr * 0.9
        local_weights, local_losses, local_models = [], [], []
        print('\n | Global Training Round : {}|\n'.format(epoch+1))
        global_model.train()
        local_model_on_new_task = copy.deepcopy(global_model)
        init_network(local_model_on_new_task)
        local_model_on_new_task.train()
        m = max(int(config.frac * config.num_users), 1)
        idxs_users = np.random.choice(range(config.num_users), m, replace=False)

        for idx in idxs_users:
            user_task = config.user_task[idx]
            user_data = config.user_data[idx]
            local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                      user_task=user_task, user_data=user_data, logger=logger,
                                      current_task=current_task, lr=lr)
            w, loss, model = local_model.update_weights_DMC_new(
                model=copy.deepcopy(local_model_on_new_task), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_models.append(copy.deepcopy(model))
        
        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        surrogate_data_ls = [surrogate_data for i in range(current_task+1)]

        if current_task > 0:
        # idxs_users = np.random.choice(range(config.num_users), int(m*0.4), replace=False)
            for i, idx in enumerate(idxs_users):
                user_task = config.user_task[idx]
                user_data = config.user_data[idx]
                local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                          user_task=user_task, user_data=user_data, logger=logger,
                                          current_task=current_task, lr=lr)
                w, loss, model = local_model.update_weights_DMC_combine(
                    global_model=copy.deepcopy(init_model), new_local_model=copy.deepcopy(global_model), global_round=epoch)
                local_weights[i] = copy.deepcopy(w)
                local_losses[i] = copy.deepcopy(loss)
                local_models[i] = copy.deepcopy(model)

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        # return
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(config.num_users):
            acc_info = []
            user_task = config.user_task[c]
            user_data = config.user_data[c]
            local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                      user_task=user_task, user_data=user_data, logger=logger,
                                      current_task=current_task, lr=lr)
            acc, loss = local_model.inference(model=global_model, idx=c)

            list_acc.append(acc)
            list_loss.append(loss)
            acc_info.append(current_task)
            acc_info.append(epoch)
            acc_info.append(c)
            acc_info.append(acc)
            train_acc_info_list.append(acc_info)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
            print('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            writer.add_scalar("loss/train_{}".format(current_task), np.mean(np.array(train_loss)), epoch)
            writer.add_scalar("acc/train_{}".format(current_task), 100*train_accuracy[-1], epoch)

        for idx in range(config.num_users):
            for i in range(config.task_number):
                if i <= current_task:
                    test_acc_info = []
                    user_task = config.user_task[idx]
                    user_data = config.user_data[idx]
                    local_model = LocalUpdate(args=config, train_data=train_dataset, test_data=dev_datasets,
                                              user_task=user_task, user_data=user_data, logger=logger,
                                              current_task=i, lr=lr)
                    test_acc, test_loss = local_model.inference(model=global_model, idx=idx)
                    print("Current Task {} Test Task {} Accuracy: {:.2f}%".format(current_task, i, 100 * test_acc))
                    test_acc_info.append(current_task)
                    test_acc_info.append(epoch)
                    test_acc_info.append(i)
                    test_acc_info.append(test_acc)
                    test_acc_info_list.append(test_acc_info)

    df1 = pd.DataFrame(train_acc_info_list, columns=['Current_task', 'Epoch', 'Client_id', 'Train_acc'])
    df2 = pd.DataFrame(test_acc_info_list, columns=['Current_task', 'Epoch', 'Test_task', 'Test_acc'])
    df1.to_csv('Results/train_acc_dmc_tasknum_{}_{}.csv'.format(config.task_number, config.model_name))
    df2.to_csv('Results/test_acc_dmc_tasknum_{}_{}.csv'.format(config.task_number, config.model_name))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


def test(config, model, test_iter, current_task):
    # test
    model.eval()
    start_time = time.time()
    subtract_val = 0
    for k in range(current_task):
        subtract_val += config.task_class_length[k]
    config.subtract_val = subtract_val
    test_acc, test_loss = evaluate(config, model, test_iter, current_task, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def evaluate(config, model, data_iter, current_task, test=False):
    acc, loss_total = test_inference(config, model, data_iter, current_task)
    return acc, loss_total / len(data_iter)
# coding: UTF-8
import time
import torch
import torch.nn as nn
import numpy as np
from train_eval_fed import train, test, train_CFeD, train_ewc, train_multihead, train_lwf, train_DMC, train_ccfed
from importlib import import_module
from utils import build_usergroup, get_parameter_number, init_network, init_network_resnet
import argparse
import copy
from utils import build_dataset, build_iterator, get_time_dif, build_dataset_from_csv_fed, build_dataset_cifar10, \
    build_dataset_mnist, build_cifar_iterator, build_dataset_cifar100

def get_args():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--model', type=str, required=False, default='CNN_Cifar10', help='choose a model: LeNet_FashionMNIST,CNN_Cifar10,CNN_Cifar100,TextCNN')
    parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
    parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
    parser.add_argument('--paradigm', default='sacfl', type=str, help='choose the training paradigm: CFeD, ewc, multihead,sacfl,dmc,lwf')
    parser.add_argument('--scenario', default='class', type=str, help=':Class-IL or Domain-IL') # class or domain
    parser.add_argument('--distribution', default=True, type=bool, help='True means iid, while False means non-iid')
    args = parser.parse_args()
    return args

args = get_args()

if __name__ == '__main__':
    print(args.paradigm)
    embedding = 'embedding.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model
    scenario = args.scenario.lower()

    x = import_module('models.' + model_name)
    config = x.Config(scenario, embedding, args.scenario)
    config.scenario = scenario
    config.paradigm = args.paradigm
    config.seed = 999
    config.iid = args.distribution
    config.model = args.model
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print(start_time)
    print("Loading data...")
    if args.model == 'CNN_Cifar100':
        vocab, train_datas, dev_datas, test_datas = build_dataset_cifar100(config)
    elif args.model == 'CNN_Cifar10':
        vocab, train_datas, dev_datas, test_datas = build_dataset_cifar10(config)
    elif 'MNIST' in args.model:
        vocab, train_datas, dev_datas, test_datas = build_dataset_mnist(config)
    else:
        vocab, train_datas, dev_datas, test_datas = build_dataset_from_csv_fed(config, args.word)

    time_dif = get_time_dif(start_time)

    # train
    config.n_vocab = len(vocab)
    config.paradigm = args.paradigm

    config.model_name = model_name
    print('config.task_number', config.task_number)
    if 'cifar' in config.model_name.lower():
        from torchvision import models
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, config.num_classes)
        model = model.to(config.device)
    else:
        model = x.Model(config).to(config.device)
        init_network(model)
    if args.paradigm.lower() == 'sacfl':
        train_ccfed(config, model, train_datas, dev_datas, 1)
    elif args.paradigm.lower() == 'cfed':
        train_CFeD(config, model, train_datas, dev_datas, copy.deepcopy(train_datas[-1]))
    elif args.paradigm.lower() == 'ewc':
        train_ewc(config, model, train_datas, dev_datas)
    elif args.paradigm.lower() == 'multihead' or model_name == 'TextCNN_multihead':
        train_multihead(config, model, train_datas, dev_datas)
    elif args.paradigm.lower() == 'dmc':
        train_DMC(config, model, train_datas, dev_datas, copy.deepcopy(train_datas[-1]))
    elif args.paradigm.lower() == 'lwf':
        train_lwf(config, model, train_datas, dev_datas)
    elif args.paradigm.lower() == 'fedprox' or 'fedavg':
        train(config, model, train_datas, dev_datas)
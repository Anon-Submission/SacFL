#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.MyModel import MyModel
from dataset import data_process


class Config(object):

    def __init__(self, scenario, embedding, scenario1):
        self.model_name = 'TextCNN'
        self.task_number = 3

        if 'class' in scenario:
            self.num_users = 50
            task_class_length, train_tasks, dev_tasks, user_data, user_task = data_process.process_text_class(self.num_users, self.task_number)
            self.task_class_length = task_class_length
            self.train_tasks = train_tasks
            self.dev_tasks = dev_tasks
            self.test_tasks = dev_tasks
            self.user_data = user_data
            self.user_task = user_task

        self.vocab_path = 'dataset/vocab.pkl'
        self.save_path = 'saved_dict/' + self.model_name + '.ckpt'
        self.log_path = 'log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('dataset/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = 0.1
        self.num_classes = 10
        self.n_vocab = 0
        self.num_epochs = 50
        self.batch_size = 64

        self.frac = 1
        self.local_ep = 5
        self.local_bs = 64
        self.iid = False
        self.server_distillation = False

        self.verbose = 1
        self.pad_size = 24
        self.learning_rate = 0.005
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256


class Model(MyModel):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc100 = nn.Linear(config.num_filters * len(config.filter_sizes), 100)
        self.fc = nn.Linear(100, config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        mid_val = self.fc100(out)
        out = self.fc(mid_val)
        return out, mid_val
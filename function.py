#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import sys
import os
import time
import csv
from networks.Networks import Networks

class Traintest():

    def __init__(self, module, device=None, network=None, loss=None, optimizer=None):
        super().__init__()
        if network is None:
            raise RuntimeError('Please pass a network as a parameter for the class ', self.__class__.__name__)
        else:
            self.network = network
        self.module = module
        self.device = device if device is not None else torch.device('cpu')
        self.loss = loss if loss is not None else torch.nn.CrossEntropyLoss().to(device)
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.4, nesterov=True)

    def train(self, trainset):
        self.network.train()
        running_loss = 0.0
        running_total = 0
        total_time_required = 0
        start = time.time()
        for input, target in trainset:
            target = torch.tensor([self.ishitormiss(target[0])]).to(self.device)
            input = input.to(self.device)
            target = target.to(self.device)
            self.network.zero_grad()
            output = self.network.forward(input)
            l = self.loss(output, target)
            #if self.module != Networks.POSITIONLSTM:
            l.backward()
            self.optimizer.step()
            del input, output
            torch.cuda.empty_cache()
            running_loss += l.item()
            running_total += target.size(0)
            end = time.time()
            time_required = (end-start)
            total_time_required += time_required
            self.__print(time_required, total_time_required, running_total, len(trainset.dataset))
            start = time.time()
        return running_loss

    def test(self, testset):
        self.network.eval()
        correct = 0
        running_total = 0
        running_loss = 0.0
        total_time_required = 0
        with torch.no_grad():
            start = time.time()
            for input, target in testset:
                target = torch.tensor([self.ishitormiss(target[0])]).to(self.device)
                input = input.to(self.device)
                target = target.to(self.device)
                self.network.zero_grad()
                output = self.network.forward(input)
                l = self.loss(output, target)
                running_loss += l.item()
                _, predicted = torch.max(output.data, 1)
                del input, output
                torch.cuda.empty_cache()
                running_total += target.size(0)
                correct += (predicted == target).sum().item()
                end = time.time()
                time_required = (end-start)
                total_time_required += time_required
                self.__print(time_required, total_time_required, running_total, len(testset.dataset))
                start = time.time()
        return correct, running_loss

    def predict(self, validationset):
        self.network.eval()
        dictpred = {}
        running_total = 0
        total_time_required = 0
        with torch.no_grad():
            start = time.time()
            for input, sample in validationset:
                target = torch.tensor(self.ishitormiss(sample[0]))
                input = input.to(self.device)
                outputs = self.network.forward(input)
                predicted = torch.argmax(outputs.data, 1)
                predicted = predicted.item()
                target = target.item()
                if predicted == 0:
                    category = 'm'
                elif predicted == 1:
                    category = 'h'
                else:
                    raise ValueError()
                dictpred[target] = category
                running_total += 1
                end = time.time()
                time_required = (end-start)
                total_time_required += time_required
                self.__print(time_required, total_time_required, running_total, len(validationset.dataset))
                start = time.time()
        return dictpred

    def __print(self, time_required, total_time_required, total, num_samples):
        outstr = 'Time required for last sample: {:.2f}sec. Total time: {:.2f}sec.  Total tests: {}/{}'.format(time_required, total_time_required, total, num_samples)
        sys.stdout.write('\r'+ outstr)
    
    def ishitormiss(self, sample):
        label = None       # it is for validation purpose, 2 indicates validation and no label is available
        if 'miss' in sample:
            label = 0
        elif 'hit' in sample:
            label = 1
        else:
            #raise ValueError('No hit or miss data found.')
            label = int(os.path.basename(sample))
        return label

class TraintestPOSITIONLSTM():
    def __init__(self, module, device=None, network=None, loss=None, optimizer=None):
        super().__init__()
        if network is None:
            raise RuntimeError('Please pass a network as a parameter for the class ', self.__class__.__name__)
        else:
            self.network = network
        self.module = module
        self.device = device if device is not None else torch.device('cpu')
        self.loss = loss if loss is not None else torch.nn.MSELoss().to(device)
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.4, nesterov=True)

    def train(self, trainset):
            self.network.train()
            running_loss = 0.0
            running_total = 0
            total_time_required = 0
            start = time.time()
            for input, target in trainset:
                target = torch.tensor([self.ishitormiss(target[0])]).to(self.device)
                input = input.to(self.device)
                self.network.zero_grad()
                output = self.network.forward(input)
                l = self.loss(output, target)
                self.optimizer.step()
                running_loss += l.item()
                del input, output
                torch.cuda.empty_cache()
                running_total += target.size(0)
                end = time.time()
                time_required = (end-start)
                total_time_required += time_required
                self.__print(time_required, total_time_required, running_total, len(trainset.dataset))
                start = time.time()
            return running_loss

    def __print(self, time_required, total_time_required, total, num_samples):
        outstr = 'Time required for last sample: {:.2f}sec. Total time: {:.2f}sec.  Total tests: {}/{}'.format(time_required, total_time_required, total, num_samples)
        sys.stdout.write('\r'+ outstr)

    def ishitormiss(self, sample):
        label = None       # it is for validation purpose, 2 indicates validation and no label is available
        if 'miss' in sample:
            label = 0
        elif 'hit' in sample:
            label = 1
        else:
            #raise ValueError('No hit or miss data found.')
            label = int(os.path.basename(sample))
        return label
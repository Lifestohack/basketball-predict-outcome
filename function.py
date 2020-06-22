#!/usr/bin/env python

import torch
import sys
import time

class Traintest():

    def __init__(self, device=None, network=None, loss=None, optimizer=None):
        super().__init__()
        if network is None:
            raise RuntimeError('Please pass a network as a parameter for the class ', self.__class__.__name__)
        else:
            self.network = network
        self.device = device if device is not None else torch.device('cpu')
        self.loss = loss if loss is not None else torch.nn.CrossEntropyLoss().to(device)
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.4, nesterov=True)

    def train(self, trainset):
        self.network.train()
        running_loss = 0.0
        running_total = 0
        total_time_required = 0
        start = time.time()
        for inputs, targets in trainset:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.network.zero_grad()
            inputs = self.__resize(inputs)
            outputs = self.network.forward(inputs)
            l = self.loss(outputs, targets)
            l.backward()
            self.optimizer.step()
            del inputs, outputs
            torch.cuda.empty_cache()
            running_loss += l.item()
            running_total += targets.size(0)
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
            for inputs, targets in testset:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.network.zero_grad()
                inputs = self.__resize(inputs)
                outputs = self.network.forward(inputs)
                l = self.loss(outputs, targets)
                running_loss += l.item()
                _, predicted = torch.max(outputs.data, 1)
                del inputs, outputs
                torch.cuda.empty_cache()
                running_total += targets.size(0)
                correct += (predicted == targets).sum().item()
                end = time.time()
                time_required = (end-start)
                total_time_required += time_required
                self.__print(time_required, total_time_required, running_total, len(testset.dataset))
                start = time.time()
        return correct, running_loss

    def predict(self, validationset):
        self.network.eval()
        prediction = []
        with torch.no_grad():
            for inputs, sample in validationset:
                dictpred = {}
                inputs = inputs.to(self.device)
                inputs = self.__resize(inputs)
                outputs = self.network.forward(inputs)
                predicted = torch.argmax(outputs.data, 1)
                predicted = predicted.item()
                sample = sample.item()
                if predicted == 0:
                    category = 'm'
                elif predicted == 1:
                    category = 'h'
                else:
                    raise ValueError()
                dictpred['id'] = sample
                dictpred['category'] = category
                prediction.append(dictpred)
        return prediction

    def __resize(self, inputs):
        if len(inputs.shape) == 7:
            return inputs.permute(0, 1, 2, 4, 3, 5, 6)
        elif  len(inputs.shape) == 6:
            return inputs.permute(0, 1, 3, 2, 4, 5)
        else:
            raise RuntimeError('Shape of the input for OPTICALCNN3D is wrong. Please check.')

    def __print(self, time_required, total_time_required, total, num_samples):
        outstr = 'Time required for last sample: {:.2f}sec. Total time: {:.2f}sec.  Total tests: {}/{}'.format(time_required, total_time_required, total, num_samples)
        sys.stdout.write('\r'+ outstr)
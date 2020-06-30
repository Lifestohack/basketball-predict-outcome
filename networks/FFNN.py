#!/usr/bin/env python3

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self, in_features, out_features, drop_p, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.drop_p = drop_p
        self.bias = bias

        if self.drop_p is None:
            self.drop_p = 0.5

        self.fcout1 = 256
        self.fcout2 = 128

        # Weights are initialized based on Kaiming Initialization.
        # weights = random weights * squarerootof(2/numberoffeature)
        # variance = 2/numberoffeatures
        self.fc1  = nn.Sequential(
            nn.Linear(self.in_features, self.fcout1, self.bias),
            nn.ReLU(inplace=True),
        )
        self.fc2  = nn.Sequential(
            nn.Linear(self.fcout1, self.fcout2, self.bias),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_p)
        )
        self.fc3  = nn.Linear(self.fcout2, self.out_features, self.bias)


    def forward(self, input):
        input = self.__resize(input)
        output = self.fc1(input)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

    def __resize(self, input):
        return input.reshape(1,-1)

#!/usr/bin/env python3

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self, in_features, out_features, drop_p, fcout, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.drop_p = drop_p
        self.fcout = fcout 
        self.bias = bias

        if self.drop_p is None:
            self.drop_p = 0.4
        if self.fcout is None:
            self.fcout = [256, 128]

        # Weights are initialized based on Kaiming Initialization.
        # weights = random weights * squarerootof(2/numberoffeature)
        # variance = 2/numberoffeatures
        self.fc1  = nn.Sequential(
            nn.Linear(self.in_features, self.fcout[0], self.bias),
            nn.ReLU(inplace=True),
        )
        self.fc2  = nn.Sequential(
            nn.Linear(self.fcout[0], self.fcout[1], self.bias),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_p)
        )
        self.fc3  = nn.Linear(self.fcout[1], self.out_features, self.bias)

    def forward(self, x):
        outputs = self.fc1(x)
        outputs = self.fc2(outputs)
        outputs = self.fc3(outputs)
        return outputs
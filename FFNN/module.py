#!/usr/bin/env python3

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self, in_features=3, out_features=2, bias=False, drop_p=0.5, fcout=[256, 128]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.drop_p = drop_p
        self.fcout = fcout 
        self.bias = bias
        self.fc1 = nn.Linear(self.in_features, self.fcout[0], bias)
        self.fc2 = nn.Linear(self.fcout[0], self.fcout[1], bias)
        self.fc3 = nn.Linear(self.fcout[1], self.out_features, bias)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(self.drop_p)

    def forward(self, x):
        outputs = F.relu(self.fc1(x))
        outputs = F.relu(self.fc2(outputs))
        outputs = self.drop(outputs)
        outputs = self.fc3(outputs)
        return outputs
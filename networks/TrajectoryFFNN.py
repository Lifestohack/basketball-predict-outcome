#!/usr/bin/env python3

import torch
import torch.nn as nn

class TrajectoryFFNN(nn.Module):
    def __init__(self, in_features, out_features, drop_p=0.5, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.drop_p = drop_p
        self.bias = bias
        self.fcout1 = in_features//2
        self.fcout2 = self.fcout1//2
        self.fcout3 = self.fcout2//2

        self.fc1  = nn.Sequential(
            nn.Linear(self.in_features, self.fcout1, self.bias),
            nn.ReLU(inplace=True),
        )
        self.fc2  = nn.Sequential(
            nn.Linear(self.fcout1, self.fcout2, self.bias),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_p)
        )
        self.fc3  = nn.Sequential(
            nn.Linear(self.fcout2, self.fcout3, self.bias),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_p)
        )
        self.fc4  = nn.Linear(self.fcout3, self.out_features, self.bias)


    def forward(self, input):
        input = self.__resize(input)
        output = self.fc1(input)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        return output

    def __resize(self, input):
        return input.reshape(1,-1)
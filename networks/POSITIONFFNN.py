#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class POSITIONFFNN(nn.Module):
    def __init__(self, num_frames, out_features):
        super().__init__()
        self.num_frames = num_frames
        self.in_features = 2 * self.num_frames * 4
        self.out_features = out_features
        self.bias = True
        self.fcout1 = self.in_features//2
        self.fcout2 = self.fcout1//2
        self.fcout3 = self.fcout2//2
        
        self.fc1  = nn.Sequential(
            nn.Linear(self.in_features, self.fcout1, self.bias),
            nn.ReLU(inplace=True)
        )
        self.fc2  = nn.Sequential(
            nn.Linear(self.fcout1, self.fcout2, self.bias),
            nn.ReLU(inplace=True)
        )
        self.fc3  = nn.Sequential(
            nn.Linear(self.fcout2, self.fcout3, self.bias),
            nn.ReLU(inplace=True)
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
        output = torch.stack([input.squeeze()[0][:self.num_frames], input.squeeze()[1][:self.num_frames]]).unsqueeze(dim=0)
        return output.view(1, -1)
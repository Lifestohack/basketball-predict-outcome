#!/usr/bin/env python3

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.linearlayer = nn.Linear(input, output)

    def forward(self, input):
        return self.linearlayer(input)

def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

class CNN(nn.Module):
    def __init__(self, img_x=48, img_y=48, in_channels=3, output=2, drop_p=0.2, fc_hidden1=256, fc_hidden2=128,t_dim=100):
        super().__init__()
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        self.in_channels = in_channels
        self.output = 2
        self.drop_p = drop_p
        self.fc_hidden1=fc_hidden1, 
        self.fc_hidden2=fc_hidden1,
        self.ch1, self.ch2, self.ch3 = 32, 48, 64
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding
        self.t_dim = t_dim

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)


        self.conv1 = nn.Conv3d(in_channels=3, out_channels=self.ch1, kernel_size=1) #Batch x Channel x Depth x Height x Width
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        

        self.conv3 = nn.Conv3d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn3 = nn.BatchNorm3d(self.ch3)


        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        #li = self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2]
        self.fc1 = nn.Linear(185856, 256)  # fully connected hidden layer
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # fully connected layer, output = multi-classes


    def forward(self, input):
        input =  input.view(1 , 3, 100, 48, 48)
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)
        return x

class LTSM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


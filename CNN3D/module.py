#!/usr/bin/env python3

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CNN3D(nn.Module):
    def __init__(self, width=48, height=48, frames=100, in_channels=3, out_features=2, drop_p=0.2, fc1out=256, fc2out=128):
        super().__init__()
        #Batch x Channel x Depth x Height x Width
        self.width = width
        self.height = height
        self.frames = frames
        self.in_channels = in_channels
        self.out_features = out_features
        self.drop_p = drop_p
        self.fc1out = fc1out 
        self.fc2out = fc2out
        self.ch1, self.ch2, self.ch3 = 32, 64, 128
        self.k1, self.k2, self.k3 = (5, 5, 5), (3, 3, 3), (1, 1, 1)      # 3d kernel size
        self.s1, self.s2, self.s3 = (2, 2, 2), (2, 2, 2), (2, 2, 2)      # 3d strides
        self.pd1, self.pd2, self.pd3 = (0, 0, 0), (0, 0, 0), (0, 0, 0)   # 3d padding

        # compute conv1, conv2, conv3 output shape
        self.conv1_outshape = self.conv3D_output_size((self.frames, self.width, self.height), self.pd1, self.k1, self.s1)
        self.conv2_outshape = self.conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = self.conv3D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        inputlinearvariables = self.ch3 * self.conv3_outshape[0] * self.conv3_outshape[1] * self.conv3_outshape[2]

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.conv3 = nn.Conv3d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3)
        self.bn3 = nn.BatchNorm3d(self.ch3)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        #self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(inputlinearvariables, self.fc1out)  # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc1out, self.fc2out)
        self.fc3 = nn.Linear(self.fc2out, self.out_features)  # fully connected layer, output = multi-classes


    def forward(self, input):
        # Conv 1
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

    def conv3D_output_size(self, img_size, padding, kernel_size, stride):
        # compute output shape of conv3D
        outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                    np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                    np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
        return outshape
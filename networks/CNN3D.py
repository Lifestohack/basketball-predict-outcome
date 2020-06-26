#!/usr/bin/env python3

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CNN3D(nn.Module):
    def __init__(self, width, height, num_frames, in_channels, out_features, fcout, drop_p=0.4):
        super().__init__()
        #Batch x Channel x Depth x Height x Width
        self.width = width
        self.height = height
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.out_features = out_features
        self.drop_p = drop_p
        self.fcout = fcout

        if self.width is None or self.height is None or self.num_frames is None  or self.in_channels is None or self.out_features is None or self.fcout is None or self.drop_p is None:
            raise RuntimeError('Please provide parameters for CNN3D')

        self.fc1out = fcout[0]
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)      # 3d kernel size
        self.s1, self.s2, self.s3, self.s4 = (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)      # 3d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)   # 3d padding

        # compute conv1, conv2, conv3 output shape
        self.conv1_outshape = self.__conv3D_output_size((self.num_frames, self.width, self.height), self.pd1, self.k1, self.s1)
        self.conv2_outshape = self.__conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = self.__conv3D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        self.conv4_outshape = self.__conv3D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)
        inputlinearvariables = self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1] * self.conv4_outshape[2] #3393024

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm3d(self.ch1),
            nn.ReLU(inplace=True),
            #nn.Dropout3d(self.drop_p) # no dropout in input layer
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm3d(self.ch2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.drop_p) # no dropout in input layer
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm3d(self.ch3),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.drop_p) # no dropout in input layer
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm3d(self.ch4),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.drop_p) # no dropout in input layer
        )

        self.fc1  = nn.Sequential(
            nn.Linear(inputlinearvariables, self.fc1out),  # fully connected hidden layer
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_p)
        )
        self.fc2  = nn.Sequential(
            nn.Linear(self.fc1out, self.fc1out),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_p)
        )
        self.fc3 = nn.Linear(self.fc1out, self.out_features)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(1, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def __conv3D_output_size(self, img_size, padding, kernel_size, stride):
        # compute output shape of conv3D
        outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                    np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                    np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
        return outshape
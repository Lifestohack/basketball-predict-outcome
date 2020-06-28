#!/usr/bin/env python3

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

class CNN3D(nn.Module):
    def __init__(self, width, height, num_frames, in_channels, out_features, drop_p=0.5):
        super().__init__()
        #Batch x Channel x Depth x Height x Width
        self.width = width
        self.height = height
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.out_features = out_features
        self.drop_p = drop_p

        if self.width is None or self.height is None or self.num_frames is None  or self.in_channels is None or self.out_features is None or self.drop_p is None:
            raise RuntimeError('Please provide parameters for CNN3D')


        self.ch1, self.ch2, self.ch3, self.ch4, self.ch5 = 16, 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 , self.k5 = (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2) , (2, 2, 2)      # 3d kernel size
        self.s1, self.s2, self.s3, self.s4 , self.s5 = (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)       # 3d strides
        self.pd1, self.pd2, self.pd3, self.pd4 , self.pd5 = (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)    # 3d padding

        # compute conv1, conv2, conv3 output shape
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm3d(self.ch1),
            nn.ReLU(inplace=True)
        )
        self.conv1_outshape = self.__conv3D_output_size((self.num_frames, self.width, self.height), self.pd1, self.k1, self.s1)
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm3d(self.ch2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.drop_p)
        )
        self.conv2_outshape = self.__conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm3d(self.ch3),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.drop_p)
        )
        self.conv3_outshape = self.__conv3D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)

        outshape = self.conv3_outshape
        channel = self.ch3
        if self.num_frames == 100 or self.num_frames == 55:
            self.conv4 = nn.Sequential(
                nn.Conv3d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
                nn.BatchNorm3d(self.ch4),
                nn.ReLU(inplace=True),
                nn.Dropout3d(self.drop_p)
            )
            self.conv4_outshape = self.__conv3D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)
            outshape = self.conv4_outshape
            channel = self.ch4
            if self.num_frames == 100:
                self.conv5 = nn.Sequential(
                    nn.Conv3d(in_channels=self.ch4, out_channels=self.ch5, kernel_size=self.k5, stride=self.s5, padding=self.pd5),
                    nn.BatchNorm3d(self.ch5),
                    nn.ReLU(inplace=True),
                    nn.Dropout3d(self.drop_p)
                )
                self.conv5_outshape = self.__conv3D_output_size(self.conv4_outshape, self.pd5, self.k5, self.s5)
                outshape = self.conv5_outshape
                channel = self.ch5
        inputlinearvariables = channel * outshape[0] * outshape[1] * outshape[2]
        self.fc1out = 1024
        self.fc2out = 512
        self.fc1  = nn.Sequential(
            nn.Linear(inputlinearvariables, self.fc1out),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_p)
        )
        self.fc2  = nn.Sequential(
            nn.Linear(self.fc1out, self.fc2out),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_p)
        )
        self.fc3 = nn.Linear(self.fc2out, self.out_features)

    def forward(self, input):
        input = self.__resize(input)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        if self.num_frames == 100 or self.num_frames == 55:
            output = self.conv4(output)
            if self.num_frames == 100:
                output = self.conv5(output)
        output = output.view(1, -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

    def __conv3D_output_size(self, img_size, padding, kernel_size, stride):
        # compute output shape of conv3D
        outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                    np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                    np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
        return outshape

    def __resize(self, input):
        output = None
        # if two views then concatenated
        if len(input.shape) == 6: 
            input = torch.cat([input[0][0], input[0][1]], dim=2).unsqueeze(dim=0)
        if len(input.shape) == 6:
            output = input.permute(0, 1, 3, 2, 4, 5)
        elif  len(input.shape) == 5:
            output = input.permute(0, 2, 1, 3, 4)
        else:
            raise RuntimeError('Shape of the input for CNN3D is wrong. Please check.')
        return output
#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class TwostreamCnn3d(nn.Module):
    def __init__(self, width, height, num_frames, in_channels, out_features=2, bias=False, drop_p=0.4, f_combine_cout=[256, 128], d_conv3d_out=[256, 128]):
        super().__init__()
        
        # Conv3d 1 starts here#
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.out_features = out_features
        self.d_conv3d_out = d_conv3d_out 
        self.drop_p = drop_p
        self.ch1, self.ch2, self.ch3 = 32, 64, 128
        self.k1, self.k2, self.k3 = (5, 5, 5), (3, 3, 3), (2, 2, 2)      # 3d kernel size
        self.s1, self.s2, self.s3 = (2, 2, 2), (2, 2, 2), (2, 2, 2)      # 3d strides
        self.pd1, self.pd2, self.pd3 = (0, 0, 0), (0, 0, 0), (0, 0, 0)   # 3d padding

        # compute conv1, conv2, conv3 output shape
        self.conv1_outshape = self.conv3D_output_size((self.num_frames, self.width, self.height), self.pd1, self.k1, self.s1)
        self.conv2_outshape = self.conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = self.conv3D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        inputlinearvariables = self.ch3 * self.conv3_outshape[0] * self.conv3_outshape[1] * self.conv3_outshape[2]

        self.conv1 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.conv3 = nn.Conv3d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3)
        self.bn3 = nn.BatchNorm3d(self.ch3)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        # Conv3d 1 end here#

        # Conv3d 2 for dense flow starts here #
        self.conv_optical_1 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1)
        self.bn_optical_1 = nn.BatchNorm3d(self.ch1)
        self.conv_optical_2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2)
        self.bn_optical_2 = nn.BatchNorm3d(self.ch2)
        self.conv_optical_3 = nn.Conv3d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3)
        self.bn_optical_3 = nn.BatchNorm3d(self.ch3)
        self.relu_optical = nn.ReLU(inplace=True)
        self.drop_optical = nn.Dropout3d(self.drop_p)
        # Conv3d 2 for dense flow end here #

        # Adding two stream of data starts here #
        self.in_features = 633600 # needs to be calculated automatically
        self.bias = bias
        self.f_combine_cout = f_combine_cout 

        self.fc1 = nn.Linear(self.in_features, self.f_combine_cout[0], bias)
        self.fc2 = nn.Linear(self.f_combine_cout[0], self.f_combine_cout[1], bias)
        self.fc3 = nn.Linear(self.f_combine_cout[1], self.out_features, bias)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(self.drop_p)
        # Adding two stream of data ends here #

    def cnn3d(self, input):
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
        return x

    def cnn3d_optical(self, input):
        # Conv 1
        x = self.conv_optical_1(input)
        x = self.bn_optical_1(x)
        x = self.relu_optical(x)
        x = self.drop_optical(x)
        # Conv 2
        x = self.conv_optical_2(x)
        x = self.bn_optical_2(x)
        x = self.relu_optical(x)
        x = self.drop_optical(x)
        # Conv 3
        x = self.conv3(x)
        x = self.bn_optical_3(x)
        x = self.relu_optical(x)
        x = self.drop_optical(x)
        return x

    def combinetwostream(self, cnn3d, optical):
        x = cnn3d.view(1, -1)
        y = optical.view(1, -1)
        inputs = torch.cat([x,y]).view(1,-1)
        outputs = F.relu(self.fc1(inputs))
        outputs = F.relu(self.fc2(outputs))
        outputs = self.drop(outputs)
        outputs = self.fc3(outputs)
        return outputs

    def conv3D_output_size(self, img_size, padding, kernel_size, stride):
        # compute output shape of conv3D
        outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                    np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                    np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
        return outshape
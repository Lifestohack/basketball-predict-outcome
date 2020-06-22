#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class TwoStream(nn.Module):
    def __init__(self, width, height, num_frames, in_channels, out_features=2, bias=False, drop_p=0.4, fc_combo_out=[1024]):
        super().__init__()
        
        # Conv3d 1 starts here#
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.out_features = out_features
        self.drop_p = drop_p
        self.ch1, self.ch2, self.ch3 = 16, 32, 64
        self.k1, self.k2, self.k3, self.k4 = (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)      # 3d kernel size
        self.s1, self.s2, self.s3, self.s4 = (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)      # 3d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)   # 3d padding


        # compute conv1, conv2, conv3 output shape
        self.conv1_outshape = self.__conv3D_output_size((self.num_frames, self.width, self.height), self.pd1, self.k1, self.s1)
        self.conv2_outshape = self.__conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = self.__conv3D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        inputlinearvariables = self.ch3 * self.conv3_outshape[0] * self.conv3_outshape[1] * self.conv3_outshape[2]

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm3d(self.ch1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.drop_p)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm3d(self.ch2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.drop_p)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm3d(self.ch3),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.drop_p)
        )
        # Conv3d 1 end here#

        # Conv3d 2 for dense flow starts here #
        self.conv_optical_1 = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm3d(self.ch1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.drop_p)
        )
        
        self.conv_optical_2 = nn.Sequential(
            nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm3d(self.ch2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.drop_p)
        )
        
        self.conv_optical_3 = nn.Sequential(
            nn.Conv3d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm3d(self.ch3),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.drop_p)
        )
        # Conv3d 2 for dense flow end here #
        
        # Adding two stream of data starts here #
        self.ch4, self.ch5, self.ch6= 128, 256, 512
        
        self.conv_combo1 = nn.Sequential(
            nn.Conv3d(in_channels=self.ch4, out_channels=self.ch5, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm3d(self.ch5),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.drop_p)
        )
        
        self.conv_combo2 = nn.Sequential(
            nn.Conv3d(in_channels=self.ch5, out_channels=self.ch6, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm3d(self.ch6),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.drop_p)
        )
        
        self.in_features = 24576    #49152         #512->24576    #256->12288 # needs to be calculated automatically
        self.bias = bias
        self.fc_combo_out = fc_combo_out
        fc_combo1_out = self.fc_combo_out[0]
        
        self.fc_combo1 = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=fc_combo1_out, bias=self.bias),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_p)
        )
        
        self.fc_combo2 = nn.Sequential(
            nn.Linear(in_features=self.ch6, out_features=self.out_features, bias=self.bias),
        )
        # Adding two stream of data ends here #

    def forward(self, inputs):
        cnn3d_out = self.__cnn3d(inputs[0][0].unsqueeze(dim=0))           #conv3d
        optical_out = self.__cnn3d_optical(inputs[0][1].unsqueeze(dim=0)) #opticalconv3d
        outputs = self.__combinetwostream(cnn3d_out, optical_out)         #combine
        return outputs

    def __cnn3d(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def __cnn3d_optical(self, x):
        x = self.conv_optical_1(x)
        x = self.conv_optical_2(x)
        x = self.conv_optical_3(x)
        return x

    def __combinetwostream(self, cnn3d, optical):
        x = torch.cat([cnn3d, optical], dim=1)
        x = self.conv_combo1(x)
        x = self.conv_combo2(x)
        x = x.view(1,-1)
        x = self.fc_combo1(x)
        x = self.fc_combo2(x)
        return x

    def __conv3D_output_size(self, img_size, padding, kernel_size, stride):
        # compute output shape of conv3D
        outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                    np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                    np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
        return outshape
#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class TwoStream(nn.Module):
    def __init__(self, width, height, num_frames, in_channels, out_features=2, bias=False, drop_p=0.5):
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

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm3d(self.ch1),
            nn.ReLU(inplace=True),
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
        
        self.conv3_outshape = self.__conv3D_output_size(self.conv2_outshape, self.pd2, self.k2, self.s2)
       
        # Conv3d 1 end here#

        # Conv3d 2 for dense flow starts here #
        self.conv_optical_1 = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm3d(self.ch1),
            nn.ReLU(inplace=True)
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
        self.conv_combo1_outshape = self.__conv3D_output_size(self.conv3_outshape, self.pd3, self.k3, self.s3)
        
        outshape = self.conv_combo1_outshape
        channel = self.ch5
        if self.num_frames == 100 or self.num_frames == 55 :
            self.conv_combo2 = nn.Sequential(
                nn.Conv3d(in_channels=self.ch5, out_channels=self.ch6, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
                nn.BatchNorm3d(self.ch6),
                nn.ReLU(inplace=True),
                nn.Dropout3d(self.drop_p)
            )
            self.conv_combo2_outshape = self.__conv3D_output_size(self.conv_combo1_outshape, self.pd3, self.k3, self.s3)            
            outshape = self.conv_combo2_outshape
            channel = self.ch6
        inputlinearvariables = channel * outshape[0] * outshape[1] * outshape[2]
        self.in_features = inputlinearvariables
        self.bias = bias
        fc_combo1_out = 512
        
        self.fc_combo1 = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=fc_combo1_out, bias=self.bias),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_p)
        )
        self.fc_combo2 = nn.Linear(in_features=fc_combo1_out, out_features=self.out_features, bias=self.bias)
        # Adding two stream of data ends here #

    def forward(self, input):
        input = self.__resize(input)
        cnn3d_out = self.__cnn3d(input[0][0].unsqueeze(dim=0))           #conv3d
        optical_out = self.__cnn3d_optical(input[0][1].unsqueeze(dim=0)) #opticalconv3d
        output = self.__combinetwostream(cnn3d_out, optical_out)         #combine
        return output

    def __cnn3d(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        return output

    def __cnn3d_optical(self, input):
        output = self.conv_optical_1(input)
        output = self.conv_optical_2(output)
        output = self.conv_optical_3(output)
        return output

    def __combinetwostream(self, cnn3d, optical):
        output = torch.cat([cnn3d, optical], dim=1)
        output = self.conv_combo1(output)
        if self.num_frames == 100  or self.num_frames == 55 :
            output = self.conv_combo2(output)
        output = output.view(1,-1)
        output = self.fc_combo1(output)
        output = self.fc_combo2(output)
        return output

    def __conv3D_output_size(self, img_size, padding, kernel_size, stride):
        # compute output shape of conv3D
        outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                    np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                    np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
        return outshape

    def __resize(self, input):
        output = None
        if len(input.shape) == 7:
            output = input.permute(0, 1, 2, 4, 3, 5, 6)
        elif  len(input.shape) == 6:
            output = input.permute(0, 1, 3, 2, 4, 5)
        else:
            raise RuntimeError('Shape of the input for Two stream is wrong. Please check.')
        return output
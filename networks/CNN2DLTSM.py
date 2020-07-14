#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CNN2DLTSM(nn.Module):
    def __init__(self, width, height, num_frames, out_features, drop_p=0.5):
        super(CNN2DLTSM, self).__init__()

        # Encoder
        self.bidirectional=True
        self.width = width
        self.height = height
        self.encoder_fcout= [512, 256]
        self.decoder_fcin = [256]
        self.hidden_size = 256
        self.num_layers=3
        
        # Shared variable
        self.num_frames = num_frames
        self.drop_p = drop_p

        if self.width is None or self.height is None or num_frames is None or out_features is None or self.drop_p is None:
            raise RuntimeError('Please provide parameters for CNN2DLSTM')

        # Encoder Conv2d Starts#
        self.ch1, self.ch2, self.ch3, self.ch4, self.ch5 = 32, 64, 128, 256, 512                   # 16, 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4, self.k5 = (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)       # 2d kernal size
        self.s1, self.s2, self.s3, self.s4, self.s5 = (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)       # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4, self.pd5 = (0, 0), (0, 0), (0, 0), (0, 0),(0, 0)   # 2d padding

        # fully connected layer hidden nodes
        self.enc_fc1out, self.enc_fc2out = self.encoder_fcout[0],  self.encoder_fcout[1]
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1),
            nn.ReLU(),                   
            #nn.MaxPool2d(kernel_size=2),
        )
        self.conv1_outshape = self.__conv2D_output_size((self.width, self.height), self.pd1, self.k1, self.s1)  # Conv1 output shape
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(),
            nn.Dropout2d(p=self.drop_p)   
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2_outshape = self.__conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm2d(self.ch3),
            nn.ReLU(),
            nn.Dropout2d(p=self.drop_p)
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv3_outshape = self.__conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm2d(self.ch4),
            nn.ReLU(),
            nn.Dropout2d(p=self.drop_p)
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv4_outshape = self.__conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch4, out_channels=self.ch5, kernel_size=self.k5, stride=self.s5, padding=self.pd5),
            nn.BatchNorm2d(self.ch5),
            nn.ReLU(),
            nn.Dropout2d(p=self.drop_p)
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv5_outshape = self.__conv2D_output_size(self.conv4_outshape, self.pd5, self.k5, self.s5)

        self.en_out = int(self.ch5 * self.conv5_outshape[0] * self.conv5_outshape[1])
        # Encoder Conv2d ends#

        # Decoder LSTM Starts #

        self.out_features = out_features
        self.de_fc1_in = self.decoder_fcin[0] * self.num_frames

        self.lstm = nn.LSTM(
            input_size=self.en_out, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True)

        self.de_fc1 = nn.Linear(self.de_fc1_in, self.out_features)

    def forward(self, input):
        input = self.__resize(input)
        output = self.__encoder(input)      # Encoder
        output = self.__decoder(output)     # Decoder
        return output

        # Decoder LSTM ends #

    def __encoder(self, input):
        output = []
        for out in input[0]:
            # CNNs
            out = out.unsqueeze(dim=0)
            out = self.conv1(out)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.conv5(out)
            out = out.view(out.size(0), -1)
            output.append(out)
        output = torch.stack(output, dim=0).transpose_(0, 1)
        return output

    def __decoder(self, input):
        # RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, 
        # possibly greately increasing memory usage. 
        # To compact weights again call flatten_parameters().
        self.lstm.flatten_parameters()
        output, (h_n, h_c) = self.lstm(input, None)  
        output = output.view(1,-1)
        output = self.de_fc1(output)
        return output

    def __conv2D_output_size(self, img_size, padding, kernel_size, stride):
        # compute output shape of conv2D
        outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                    np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
        return outshape
        
    def __resize(self, input):
        output = None
        # if two views then concatenate them. Need to check if this is also available for other networks
        if len(input.shape) == 6:                  
            outputs = torch.cat([input[0][0], input[0][1]], dim=2).unsqueeze(dim=0) #concatenate two view
        else:
            output = input
        return output
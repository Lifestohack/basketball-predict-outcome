#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CNN2D(nn.Module):
    def __init__(self, width=48, height=48, fc1out=256, fc2out=128, drop_p=0.2, frames=100):
        super(CNN2D, self).__init__()

        self.width = width
        self.height = height
        self.frames = frames


        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = self.conv2D_output_size((self.width, self.height), self.pd1, self.k1, self.s1)  # Conv1 output shape
        self.conv2_outshape = self.conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = self.conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        self.conv4_outshape = self.conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)

        # fully connected layer hidden nodes
        self.fc1out, self.fc2out = fc1out, fc2out
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1),
            nn.ReLU(),                      
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm2d(self.ch3),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm2d(self.ch4),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.drop = nn.Dropout2d(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1], self.fc1out)   # fully connected layer, output k classes
        self.fc2 = nn.Linear(self.fc1out, self.fc2out)
        self.fc3 = nn.Linear(self.fc2out, self.frames)   # output = CNN embedding latent variables

    def forward(self, x):
        out = []
        for y in x[0]:
            # CNNs
            y = y.unsqueeze(dim=0)
            y = self.conv1(y)
            y = self.conv2(y)
            y = self.conv3(y)
            y = self.conv4(y)
            y = y.view(y.size(0), -1)           # flatten the output of conv

            # FC layers
            y = F.relu(self.fc1(y))
            y = F.dropout(y, p=self.drop_p, training=self.training)
            y = F.relu(self.fc2(y))
            y = F.dropout(y, p=self.drop_p, training=self.training)
            y = self.fc3(y)
            out.append(y)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        out = torch.stack(out, dim=0).transpose_(0, 1)
        return out

    def conv2D_output_size(self, img_size, padding, kernel_size, stride):
        # compute output shape of conv2D
        outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                    np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
        return outshape



class LTSM(nn.Module):
    def __init__(self, frames=100, num_layers=3, hidden_size=256, fc1out=128, drop_p=0.2, out_features=2, bidirectional=True):
        super(LTSM, self).__init__()
        self.frames = frames
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.drop_p = drop_p
        self.out_features = out_features
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.fc1in = 2*hidden_size # bidirectional doubles the parameter 
        self.fc1in = self.fc1in * frames
        self.fc1out = fc1out
        self.lstm = nn.LSTM(
            input_size=self.frames, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True,
            bidirectional =self.bidirectional)
        self.fc1 = nn.Linear(self.fc1in, self.fc1out)
        self.fc2 = nn.Linear(self.fc1out, self.out_features)
       

    def forward(self, x):
        # RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, 
        # possibly greately increasing memory usage. 
        # To compact weights again call flatten_parameters().
        self.lstm.flatten_parameters()
        RNN_out, (h_n, h_c) = self.lstm(x, None)  
        y = RNN_out.view(1,-1)
        x = self.fc1(RNN_out.view(1,-1))   # choose RNN_out at the last time step
        x = self.fc2(x.view(1,-1))   # choose RNN_out at the last time step
        return x
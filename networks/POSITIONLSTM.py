#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as numpy

class POSITIONLSTM(nn.Module):
    def __init__(self, num_frames, in_features):
        super().__init__()
        self.in_features = 4
        self.num_frames  = num_frames
        self.num_layers = 2
        self.out_features = 2
        self.hidden_layer_size = 8
        self.max_frames = 99
        self.window = 12

        self.prediction_linear_input = self.window * self.hidden_layer_size
        self.linear_input = 2 * self.max_frames * self.in_features
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss = torch.nn.MSELoss().to(self.device)
        self.evaluate = False

        if self.num_frames < self.max_frames:
            self.predicpredictnextframelstm_view1 = nn.LSTM(
                input_size= self.in_features,
                hidden_size=self.hidden_layer_size, 
                num_layers=self.num_layers
            )
            self.predicpredictnextframelinear_view1 = nn.Linear(self.prediction_linear_input, self.in_features)

            self.predicpredictnextframelstm_view2 = nn.LSTM(
                input_size= self.in_features,
                hidden_size=self.hidden_layer_size, 
                num_layers=self.num_layers
            )
            self.predicpredictnextframelinear_view2 = nn.Linear(self.prediction_linear_input, self.in_features)

        self.predicpredictnextframelstm_view2 = nn.LSTM(
                input_size= self.in_features,
                hidden_size=self.hidden_layer_size, 
                num_layers=self.num_layers
        )

        self.lstm1 = nn.LSTM(
                input_size= self.in_features,
                hidden_size=self.hidden_layer_size, 
                num_layers=self.num_layers
        )
        self.fcout1 = self.linear_input//2
        self.fcout2 = self.fcout1//2
        self.fcout3 = self.fcout2//2
        self.bias = True
        self.fc1  = nn.Sequential(
            nn.Linear(self.linear_input, self.fcout1, self.bias),
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
        input = input.squeeze()
        available_frames_view1 = input[0][:self.num_frames].unsqueeze(dim=0)
        available_frames_view2 = input[1][:self.num_frames].unsqueeze(dim=0)
        if self.num_frames < self.max_frames:
            hidden = self.init_hidden(self.window)
            hidden_view1 = self.__repackage_hidden(hidden)
            hidden_view2 = self.__repackage_hidden(hidden)
            framestopredict = self.max_frames - self.num_frames
            for topredictframe in range(self.num_frames, self.max_frames):
                target_view1 = input[0][topredictframe].unsqueeze(dim=0)
                target_view2 = input[1][topredictframe].unsqueeze(dim=0)
                windowed_frames_view1 = self.getinputwindowed(available_frames_view1, topredictframe, self.window)
                windowed_frames_view2 = self.getinputwindowed(available_frames_view2, topredictframe, self.window)
                predictinput_view1, hidden_view1 = self.predicpredictnextframelstm_view1(windowed_frames_view1, hidden_view1)
                predictinput_view2, hidden_view2 = self.predicpredictnextframelstm_view1(windowed_frames_view2, hidden_view2)
                predictinput_view1 = predictinput_view1.view(1, -1)
                predictinput_view2 = predictinput_view2.view(1, -1)
                predictinput_view1 = self.predicpredictnextframelinear_view1(predictinput_view1)
                predictinput_view2 = self.predicpredictnextframelinear_view2(predictinput_view2)
                if self.training == True:
                    l_view1 = self.loss(predictinput_view1, target_view1)
                    l_view2 = self.loss(predictinput_view2, target_view2)
                    loss = l_view1 + l_view2
                    l_view1.backward(retain_graph=True)
                    l_view2.backward(retain_graph=True)
                available_frames_view1 = torch.cat([available_frames_view1.squeeze(), predictinput_view1]).unsqueeze(dim=0)
                available_frames_view2 = torch.cat([available_frames_view2.squeeze(), predictinput_view2]).unsqueeze(dim=0)
            input  = torch.cat([available_frames_view1, available_frames_view2])
        #a = input.squeeze().detach().cpu().numpy()
        #numpy.savetxt("foo1.csv", a[0], delimiter=",")
        #numpy.savetxt("foo2.csv", a[1], delimiter=",")
        input = self.fc1(input.view(1,-1))
        output = self.fc2(input)
        output = self.fc3(output)
        output = self.fc4(output)
        return output

    def __repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.__repackage_hidden(v) for v in h)

    def __resize(self, input):
        output = input.unsqueeze(dim=0)
        return output

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.hidden_layer_size).to(self.device),
                weight.new_zeros(self.num_layers, bsz, self.hidden_layer_size).to(self.device))
    
    def getinputwindowed(self, input, topredictframe, window):
        startingframe = topredictframe - window
        return input.squeeze()[startingframe:startingframe+window].unsqueeze(dim=0)
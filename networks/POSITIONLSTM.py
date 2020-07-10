#!/usr/bin/env python3

import torch
import torch.nn as nn

class POSITIONLSTM(nn.Module):
    # input_size = 4 # frame number, x, y, radius <- four values
    # output_size = 4 also predicted output the same size
    def __init__(self, num_frames, input_size=4, hidden_layer_size=100, output_size=4):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.device = torch.device('cuda:0')
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).to(self.device),
                            torch.zeros(1,1,self.hidden_layer_size).to(self.device))
        self.loss = torch.nn.MSELoss().to(self.device)
        self.optimize


    def forward(self, sample):
        # sample is the position of the ball like frame number, x, y , radius of ball.
        # that is why the the input_size of lstm is 4
        # now just working with view1. view2 ignored, but will consider next
        total_loss = torch.tensor(0).to(self.device)
        num_frames_to_predict = len(sample) - self.num_frames
        available_view = sample[:self.num_frames]   #just use the 30 frames to predict the remaining frames
        next_frame_number = self.num_frames
        for i in range(num_frames_to_predict):
            target = sample[next_frame_number].unsqueeze(dim=0)
            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(self.device),
                        torch.zeros(1, 1, self.hidden_layer_size).to(self.device))
            lstm_out, self.hidden_cell = self.lstm(available_view.view(len(available_view) ,1, -1), self.hidden_cell)
            predictions = self.linear(lstm_out.view(len(available_view), -1))
            frame_predicted = predictions[-1].unsqueeze(dim=0)
            available_view = torch.cat((available_view, frame_predicted))
            total_loss = total_loss + self.loss(frame_predicted, target)  #adding the predicted frame to the available frames and in next loop it will be used to predict again the next frame
            next_frame_number += 1
        total_loss.backward()
        # now optimizer can be used optimizer.step()
        # now as we have all the information needed till end of the frame
        # we can use a classifier to tell if hit or miss
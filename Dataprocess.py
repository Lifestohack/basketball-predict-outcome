#!/usr/bin/env python3

import torch

class Preprocess:
    
    def background_subtractor(self, tensor):
        return self.meidan(tensor)

    def meidan(self, tensor):
        background, _ = torch.median(tensor, dim=0, keepdim=True)
        ball_mask = (background - tensor).abs() > (20/255)
        ball = torch.where(ball_mask, tensor, torch.zeros_like(tensor))
        return ball
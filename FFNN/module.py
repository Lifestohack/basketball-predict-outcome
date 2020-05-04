#!/usr/bin/env python3

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linearlayer = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        return self.linearlayer(x)
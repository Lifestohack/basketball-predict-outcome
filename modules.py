#!/usr/bin/env python3

import numpy as np
import torch.nn as nn
from pathlib import Path
import torch


class FFNN(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.linearlayer = torch.nn.Linear(input, output)

    def forward(self, input):
        return self.linearlayer(input)
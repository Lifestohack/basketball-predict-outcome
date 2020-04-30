#!/usr/bin/env python3

import numpy as np
import torch.nn as nn
from pathlib import Path


class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        nn.Linear(48, 2)

    def forward(self, input):
        return input
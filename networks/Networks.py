#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum
import os

class Networks(Enum):
    FFNN = 1
    CNN3D = 2
    CNN2DLSTM = 3
    TWOSTREAM = 4
    POSITIONFFNN = 5
    POSITIONLSTM = 6
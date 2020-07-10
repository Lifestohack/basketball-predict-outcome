#!/usr/bin/env python
from enum import Enum

import os
class Networks(Enum):
    FFNN = 1
    CNN3D = 2
    CNN2DLSTM = 3
    TWOSTREAM = 4
    TRAJECTORYFFNN = 5
    TRAJECTORYLSTM = 6
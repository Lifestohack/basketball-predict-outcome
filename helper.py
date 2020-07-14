#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

def get_all_files(data_path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(data_path):
        for file in f:
            files.append(os.path.join(r, file))
    return files
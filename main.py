#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Diwas Bhattarai
# Email: diwas@bhattarai.de
# Date: 18.07.2020

import Basketball
import time
import gc
from networks.Networks import Networks

# *********** <Dataset Preprocess> ******************

# Use DataMultiProcess to preprocess the dataset
# Without those data this file doesnot work

# *********** </Dataset Preprocess> ******************

# *************<IMPORTANT>*********************
# Following points are important.

# Python memory management doesnot let explicitly freeing memory. 
# So even if loaded samples are not needed and therefore can be deleted, memory will not free up. So reading new samples will be slower as  
# memory is still being used by old and garbage value. In Basketball class destroycache() function is implemented
# to delete garbage memory but in python there is no guarantee when this will happen. 
# So please take care of following points

# After background=True dataset is used and before background=False please clear memory otherwise 
# background=True frames will be returned from cache instead of background=False frames.

# After training or validating one network please clear memory before training or validating next network. 
# If using Jupyter Notebook then by restarting.

# After training for 30 frames don't train higher than 30 frames but only lower frames.
# So always train higher frames first and then lower frames

# *************</IMPORTANT>*********************

start_time = time.time()

# **********************<Variables>**********************************
Epocs = 40

# size of the image that will be feed into the network
width = 128
height = 128

# Size for Feed forward neural network as it takes lots of resources
# May be it has lower size than the normal
width_ffnn=48
height_ffnn=48

# [training and test] or validating??
# possible values = training, validation
split = 'training'
print("**************************** {} STARTED ******************************************".format(split.upper()))
# the network you want to use
# Options are 
# FFNN, CNN3D, CNN2DLSTM, TWOSTREAM uses images 
# POSITIONFFNN, POSITIONLSTM uses CSV file
network = Networks.POSITIONLSTM

# Two dataset options are available. 
# One with background and one without background
# background = False means with no background
background = False

# if only validation is required using pretrained network then use pretrained=True and provide pretrainedpath. 
# If pretrainedpath is not provided then the latest trained network will be used.
pretrained = False

if network == Networks.FFNN:
    if width_ffnn is not None and height_ffnn is not None:
        width=width_ffnn
        height = height_ffnn

# Create first the Basketball object

# If using FFNN, CNN3D, CNN2DLSTM, TWOSTREAM
# Images will be used. First of original dataset should be preprocessed.
# Use DataMultiProcess to preprocess the dataset
#dp = Basketball.Basketball(width=width, height=height, split=split)

# If using POSITIONLSTM and POSITIONFFNN 
# Co-ordinate of basketball is used instead of images.
# For this please use balldetection.m file. 
# Open the file in matlab and give the path to the original dataset. 
# The output of the matlab file, please copy it to the dataset folder of this project under "trajectory" folder
trajectory = True if (network==Networks.POSITIONFFNN or  network==Networks.POSITIONLSTM) else False
if trajectory == True and (network != Networks.POSITIONFFNN and network != Networks.POSITIONLSTM):
    raise ValueError("{} doesnot support position values. Use {} or {} instead.".format(network, Networks.POSITIONFFNN, Networks.POSITIONLSTM))
dp = Basketball.Basketball(width=width, height=height, split=split, trajectory=trajectory)

# **************************</Variables>******************************

#*********************<Train or validate>****************************

# Maximum number of frames that are available per sample
max_frames = 100

if (background == False and max_frames > 99) or trajectory == True:
    print("For dataset where background has been removed, maximum of 99 frames are available but you requested {} frames. Using 99 frames instead.".format(max_frames))
    max_frames = 99

# learning rate is different for different networks
# learning rate between 0.1 and 0.0000001 have been tested for each network.
lr = 0.0001
if network == Networks.FFNN:
    lr = 0.000001
elif network == Networks.CNN3D:
    lr=0.0001
elif network == Networks.CNN2DLSTM:
    lr=0.0001
    if background == True:
        lr=0.000001
elif network == Networks.TWOSTREAM:
    lr=0.0001
elif network == Networks.POSITIONFFNN:
    lr=0.00001
elif network == Networks.POSITIONLSTM:
    lr=0.00001

# if after every training, testing should be done then use
testeverytrain=True

# Start training or validating
dp.run(max_frames, network, testeverytrain=testeverytrain, EPOCHS=Epocs, lr=lr, background=background,  pretrained=pretrained)
dp.run(55, network, testeverytrain=testeverytrain, EPOCHS=Epocs, lr=lr, background=background, pretrained=pretrained)
dp.run(30, network, testeverytrain=testeverytrain, EPOCHS=Epocs, lr=lr, background=background,  pretrained=pretrained)
#*********************</Train or validate>****************************

# dp.destroycache()
# del dp
# gc.collect()
end_time = time.time()
print("Total time required: {} seconds".format(end_time - start_time))
print("**************************** {} FINISHED ******************************************".format(split.upper()))

#!/usr/bin/env python

import Basketball
import time
import gc
from networks.Networks import Networks
# *************IMPORTANT*********************
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

# After training for 30 frames don't train for higher than 30 frames but only lower frames.
# So always train higher frames first and then lower frames

# *************IMPORTANT*********************

start_time = time.time()

# **********************Variables**********************************

# path to the dataset
data = "dataset"

Epocs = 20

# size of the image that will be feed into the network
width = 128
height = 128

width_ffnn=48
height_ffnn=48

# [training and test] or validating??
# possible values = validation, training
split = 'training'

# the network you want to use
# Options are FFNN, CNN3D, CNN2DLSTM, TWOSTREAM and TRAJECTORYLSTM
network = Networks.CNN2DLSTM

if network == Networks.FFNN:
    width=width_ffnn
    height = height_ffnn

# Create first the Basketball object
dp = Basketball.Basketball(width=width, height=height, data=data, split=split)

# Using co-ordinate of basketball instead of images
# For this please use balldetection.m file. 
# Open the file in matlab and give the path to the original dataset. 
# The output of the matlab file, please copy it to the dataset folder of this project under "trajectory" folder
# if network == Networks.TRAJECTORYLSTM:
#   trajectory = True
# dp = Basketball.Basketball(width=width, height=height, split=split, trajectory=True)

# **************************Variables******************************

#*********************Train or validate****************************
# Two dataset options are available. 
# One with background and one without background
background = False  # No background

# Maximum number of frames that are available per sample
max_frames = 100

if background == False and max_frames > 99:
    print("For dataset where background has been removed maximum of 99 frames are available. Using 99 frames instead.")
    max_frames = 99

# learning rate is different for different networks
lr = 0.0001
if network == Networks.FFNN:
    lr = 0.000001
elif network == Networks.CNN3D:
    lr=0.0001
elif network == Networks.CNN2DLSTM:
    lr=0.0001 
elif network == Networks.TWOSTREAM:
    lr=0.0001

# if after every training, if testing should be done then use
testeverytrain=True

# Start training or validating
dp.run(max_frames, network, testeverytrain=testeverytrain, EPOCHS=Epocs, lr=lr, background=background)
dp.run(55, network, testeverytrain=testeverytrain, EPOCHS=Epocs, lr=lr, background=background)
dp.run(30, network, testeverytrain=testeverytrain, EPOCHS=Epocs, lr=lr, background=background)

# if only validation is required using pretrained network then
#pretrained = True 
#pretrainedpath = "output/network/prediction_0.55_1_100_0.0001_CNN3D_2020_07_06_05_47_25.pt"
#dp.run(30, network, testeverytrain=True, EPOCHS=Epocs, lr=lr, background=background, pretrained=pretrained, pretrainedpath=pretrainedpath) 
#*********************Train or validate****************************

# dp.destroycache()
# del dp
# gc.collect()
end_time = time.time()
print("Total time required: {} seconds".format(end_time - start_time))
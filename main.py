#!/usr/bin/env python

import Basketball
import time
import gc
from networks.Networks import Networks
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
Epocs = 20

# size of the image that will be feed into the network
width = 128
height = 128

# Size for Feed forward neural network as it takes lots of resources
# May be it has lower size than the normal
width_ffnn=48
height_ffnn=48

# [training and test] or validating??
# possible values = validation, training
split = 'training'
print("**************************** {} STARTED ******************************************".format(split.upper()))
# the network you want to use
# Options are FFNN, CNN3D, CNN2DLSTM, TWOSTREAM and TRAJECTORYLSTM
network = Networks.TWOSTREAM

if network == Networks.FFNN:
    if width_ffnn is not None and height_ffnn is not None:
        width=width_ffnn
        height = height_ffnn

# Create first the Basketball object
dp = Basketball.Basketball(width=width, height=height, split=split)

# Using co-ordinate of basketball instead of images.
# For this please use balldetection.m file. 
# Open the file in matlab and give the path to the original dataset. 
# The output of the matlab file, please copy it to the dataset folder of this project under "trajectory" folder
# if network == Networks.TRAJECTORYLSTM:
#   trajectory = True
# dp = Basketball.Basketball(width=width, height=height, split=split, trajectory=True)

# **************************</Variables>******************************

#*********************<Train or validate>****************************
# Two dataset options are available. 
# One with background and one without background
background = False

# Maximum number of frames that are available per sample
max_frames = 100

if background == False and max_frames > 99:
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
elif network == Networks.TRAJECTORYLSTM:
    lr=0.0001

# if after every training, testing should be done then use
testeverytrain=True

# Start training or validating
dp.run(max_frames, network, testeverytrain=testeverytrain, EPOCHS=Epocs, lr=lr, background=background)
dp.run(55, network, testeverytrain=testeverytrain, EPOCHS=Epocs, lr=lr, background=background)
dp.run(30, network, testeverytrain=testeverytrain, EPOCHS=Epocs, lr=lr, background=background)

# if only validation is required using pretrained network then
#split = 'validation'
#pretrained = True 
#pretrainedpath = "output/network/prediction_0.55_1_100_0.0001_CNN3D_2020_07_06_05_47_25.pt"
#dp.run(30, network, testeverytrain=True, EPOCHS=Epocs, lr=lr, background=background, pretrained=pretrained, pretrainedpath=pretrainedpath) 
#*********************</Train or validate>****************************

# dp.destroycache()
# del dp
# gc.collect()
end_time = time.time()
print("Total time required: {} seconds".format(end_time - start_time))
print("**************************** {} FINISHED ******************************************".format(split.upper()))
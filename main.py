#!/usr/bin/env python

import Basketball
import time
import gc

# *************IMPORTANT*********************
# Following points are important.

# Python memory management doesnot let explicitly freeing memory. 
# So even if loaded samples are not needed and therefore can be deleted, memory will not free up. So reading new samples will be slower as  
# memory is still being used by old and garbage value. In Basketball class destroycache() function is implemented
# to delete garbage memory but in python there is no guarantee when this will happen. 
# So please take care of following points

# After background=True dataset is used then before background=False please clear memory otherwise 
# background=True frames will be returned from cache instead of background=False frames.

# After training or validating one network please clear memory before training or validating next network. 
# If using Jupyter Notebook then by restarting.

# After training for 30 frames don't train for higher than 30 frames but only lower frames.
# So always train higher frames first and then lower frames

# *************IMPORTANT*********************

start_time = time.time()
split = 'validation' #validation, training
network = "CNN3D"
background = True
pretrained = True
pretrainedpath = "output/network/prediction_0.55_1_100_0.0001_CNN3D_2020_07_06_05_47_25.pt"
dp = Basketball.Basketball(width=128, height=128, split=split, trajectory=False)
dp.run(100, network, testeverytrain=True, EPOCHS=1, lr=0.0001, background=background, pretrained=pretrained, pretrainedpath=pretrainedpath)
#dp.run('FFNN', testeverytrain=True, EPOCHS=EPOCHS)
#dp.run('CNN3D', testeverytrain=True, EPOCHS=EPOCHS)
#dp.run('CNN2DLSTM', testeverytrain=True, EPOCHS=EPOCHS)
#dp.run('TWOSTREAM', testeverytrain=True, EPOCHS=EPOCHS)
dp.destroycache()
del dp
gc.collect()
end_time = time.time()
print("Total time required: {} seconds".format(end_time - start_time))
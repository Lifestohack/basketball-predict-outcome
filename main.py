#!/usr/bin/env python

import Basketball
import time

start_time = time.time()
data = 'dataset\data\cropped_samples'
num_frames = 100
split = 'training' #validation, training
EPOCHS = 1

dp = Basketball.Basketball(data, width=128, height=128,  num_frames=num_frames, split=split)
#dp.run('FFNN', testeverytrain=True, EPOCHS=EPOCHS)
dp.run('CNN3D', testeverytrain=True, EPOCHS=EPOCHS)
dp.run('CNN2DLSTM', testeverytrain=True, EPOCHS=EPOCHS)
#dp.run('TWOSTREAM', testeverytrain=True, EPOCHS=EPOCHS)
end_time = time.time()
print("Total time required: {} seconds".format(end_time - start_time))
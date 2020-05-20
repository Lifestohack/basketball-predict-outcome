#!/usr/bin/env python

import Basketball
import serialize
import time

start_time = time.time()
data = 'data'
data48x48 = 'cache'
num_frames = 96
split = 'training'
EPOCHS = 10
optics = 'optics'

dp = Basketball.Basketball(data, width=128, height=128,  num_frames=num_frames, split=split)
#dp.run('FFNN', testeverytrain=True, EPOCHS=EPOCHS)
dp.run('CNN3D', testeverytrain=True, EPOCHS=EPOCHS) #may run on 4gb gpu with 32 Gb RAM
#dp.run('CNN2DLSTM', testeverytrain=True, EPOCHS=EPOCHS) #may run on 4gb gpu with 32 Gb RAM
#dp.run('OPTICALCONV3D', testeverytrain=True, EPOCHS=EPOCHS)
end_time = time.time()
print("Total time required: {} seconds".format(end_time - start_time))
#!/usr/bin/env python

import Basketball
import time

# After trainig for 30 frames don't train for higher frames but only first lower frames.
# So always train higher frames first and then lower frames
start_time = time.time()
data = 'dataset/background/data'    # format should be  "folder1/folder2/folder3"
split = 'training' #validation, training
dp = Basketball.Basketball(data, width=128, height=128, split=split, trajectory=False)

EPOCHS = 50
num_frames = 100
dp.run(num_frames, 'CNN3D', testeverytrain=True, EPOCHS=EPOCHS)

EPOCHS = 50
num_frames = 55
dp.run(num_frames, 'CNN3D', testeverytrain=True, EPOCHS=EPOCHS)

EPOCHS = 50
num_frames = 30
dp.run(num_frames, 'CNN3D', testeverytrain=True, EPOCHS=EPOCHS)

#dp.run('FFNN', testeverytrain=True, EPOCHS=EPOCHS)
#dp.run('CNN3D', testeverytrain=True, EPOCHS=EPOCHS)
#dp.run('CNN2DLSTM', testeverytrain=True, EPOCHS=EPOCHS)
#dp.run('TWOSTREAM', testeverytrain=True, EPOCHS=EPOCHS)
end_time = time.time()
print("Total time required: {} seconds".format(end_time - start_time))
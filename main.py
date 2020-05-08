import Basketball

dp = Basketball.Basketball('cache', 100, None)
#dp.run('FFNN', testeverytrain=True, EPOCHS=10)
#dp.run('CNN3D', testeverytrain=True, EPOCHS=10)
#dp.run('CNN2DLSTM', testeverytrain=True, EPOCHS=10)
dp.run('OPTICALCONV3D', testeverytrain=True, EPOCHS=1, opticalpath='opticalflow')
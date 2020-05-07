import random
import torch
import torchvision
from Dataset import Basketball
from Dataprocess import Preprocess
from torch.utils.data import DataLoader
from PIL import Image
import CNN3D.module
import CNN3D.function 
import FFNN.module 
import FFNN.function
import CNN2DLSTM.module
import CNN2DLSTM.function
import OPTICALCONV3D.module
import OPTICALCONV3D.function

img_transformation = torchvision.transforms.Compose([
    torchvision.transforms.Resize((48,48)),
    torchvision.transforms.CenterCrop((48,48)),
    torchvision.transforms.ToTensor()
])

dataprocess = Preprocess().background_subtractor


path = "data"
dataset = Basketball(path, split='training', num_frame = 100, img_transform = img_transformation, 
                     dataprocess=dataprocess, cacheifavailable=True, savecache=True, combineview=True, cachepath='cache')
trainset, testset = dataset.train_test_split(train_size=0.8)

trainset = DataLoader(trainset, shuffle=True)
testset = DataLoader(testset, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def run(module='FFNN', testeverytrain=True, EPOCHS=1):
    loss = torch.nn.CrossEntropyLoss().to(device)
    if module=='FFNN':
        out_features = 2 # only 2 classifier hit or miss
        in_features = dataset[0][0].numel()
        network = FFNN.module.FFNN(in_features=in_features, out_features=out_features)
        optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.4, nesterov=True)
        obj = FFNN.function.FFNNTraintest(device, network, loss, optimizer)
    elif module=='CNN3D':
        network = CNN3D.module.CNN3D(width=2*48, height=48, in_channels=3, out_features=2, drop_p=0.2, fc1out=256, fc2out=128, frames=100) # the shape of input will be Batch x Channel x Depth x Height x Width
        optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.4, nesterov=True)
        obj = CNN3D.function.CNN3DTraintest(device, network, loss, optimizer)
    elif module=='CNN2DLSTM':
        encoder2Dnet = CNN2DLSTM.module.CNN2D(width=2*48, height=48, drop_p=0.2, fc1out=256, fc2out=128, frames=100) # Batch x Depth x Channel x Height x Width
        decoderltsm = CNN2DLSTM.module.LTSM()
        cnn_params = list(encoder2Dnet.parameters()) + list(decoderltsm.parameters())
        optimizer = torch.optim.SGD(cnn_params, lr=0.001, momentum=0.4, nesterov=True)
        #optimizer = torch.optim.Adam(cnn_params, lr=0.0001)
        obj = CNN2DLSTM.function.CNN2DLSTMTraintest(device, encoder2Dnet, decoderltsm, loss, optimizer)
    elif module=='OPTICALCONV3D':
        cnn3d = OPTICALCONV3D.module.CNN3D(width=2*48, height=48, in_channels=3, out_features=2, drop_p=0.2, fc1out=256, fc2out=128, frames=100) # the shape of input will be Batch x Channel x Depth x Height x Width
        opticalcnn3d = OPTICALCONV3D.module.CNN3D(width=2*48, height=48, in_channels=3, out_features=2, drop_p=0.2, fc1out=256, fc2out=128, frames=100) # the shape of input will be Batch x Channel x Depth x Height x Width
        twostream = OPTICALCONV3D.module.TwostreamConv3d()
        cnn_params = list(cnn3d.parameters()) + list(opticalcnn3d.parameters()) + list(twostream.parameters())
        optimizer = torch.optim.SGD(cnn_params, lr=0.001, momentum=0.4, nesterov=True)
        obj = OPTICALCONV3D.function.OPTICALCONV3DTraintest(device, cnn3d, opticalcnn3d, twostream, loss, optimizer)
    else:
        ValueError()

    print("------------------------------------------------------")
    print("|    EPOCHS  |   Total   |   Loss    |   Accuracy    |")
    print("------------------------------------------------------")
    i = 1
    for epoch in range(0, EPOCHS):
        total, running_loss = obj.train(trainset)
        print("|      {}     |   {}     |   {:.4f}  |               |".format(i, total, running_loss/total) )
        print("------------------------------------------------------")
        if  (i == EPOCHS) or testeverytrain:
            total, correct = obj.test(testset)
            print("|      {}     |    {}     |           | ({}) {:.1f} %   |".format(i, total, correct, 100 * correct/total))
            print("------------------------------------------------------")
        i += 1
#run('FFNN', testeverytrain=True, EPOCHS=10)
#run('CNN3D', testeverytrain=True, EPOCHS=10)
#run('CNN2DLSTM', testeverytrain=True, EPOCHS=10)
run('OPTICALCONV3D', testeverytrain=True, EPOCHS=10)
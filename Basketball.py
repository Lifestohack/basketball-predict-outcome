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

img_transformation = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64,48)),
    torchvision.transforms.CenterCrop((48,48)),
    torchvision.transforms.ToTensor()
])

dataprocess = Preprocess().background_subtractor

path = "data"
dataset = Basketball(path, split='training', num_frame = 100, img_transform = img_transformation, dataprocess=dataprocess)
trainset, testset = dataset.train_test_split()

trainset = DataLoader(trainset, shuffle=True)
testset = DataLoader(testset, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def run(module='FFNN', testeverytrain=True, EPOCHS=1):
    loss = torch.nn.CrossEntropyLoss().to(device)
    if module=='FFNN':
        out_features = 2 # only 2 classifier hit or miss
        in_features = out_features*dataset[0][0][0].numel()
        network = FFNN.module.FFNN(in_features=in_features, out_features=out_features)
        optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.4, nesterov=True)
        obj = FFNN.function.FFNNTraintest(device, network, loss, optimizer)
    elif module=='CNN3D':
        network = CNN3D.module.CNN3D(width=2*48, height=48, in_channels=3, out_features=2, drop_p=0.2, fc_hidden1=256, fc_hidden2=128, frames=100) # the shape of input will be Batch x Channel x Depth x Height x Width
        optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.4, nesterov=True)
        obj = CNN3D.function.CNN3DTraintest(device, network, loss, optimizer)
    else:
        ValueError()
    network = network.to(device)
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

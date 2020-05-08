import random
import torch
import torchvision
import Dataset
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
import copy
import torch.nn as nn

class Basketball():
    def __init__(self, data, num_frame):
        super().__init__()
        if not data or not num_frame:
            raise RuntimeError('Please pass parameter.')
        self.data = data
        self.num_frame = num_frame

        dataset = Dataset.Basketball(self.data, split='training', num_frame = self.num_frame)
        self.trainset, self.testset = dataset.train_test_split(train_size=0.8)
        self.trainset_loader = DataLoader(self.trainset, shuffle=True)
        self.testset_loader = DataLoader(self.testset, shuffle=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

    def run(self, module='FFNN', testeverytrain=True, EPOCHS=1, opticalpath=None):
        if module=='FFNN':
            obj = self.__FFNN()
        elif module=='CNN3D':
            obj = self.__CNN3D()
        elif module=='CNN2DLSTM':
            obj = self.__CNN2DLSTM()
        elif module=='OPTICALCONV3D':
            obj, train_optical_loader, test_optical_loader = self.__OPTICALCONV3D(opticalpath)
        else:
            ValueError()

        print("------------------------------------------------------")
        print("|    EPOCHS  |   Total   |   Loss    |   Accuracy    |")
        print("------------------------------------------------------")
        i = 1
        for epoch in range(0, EPOCHS):
            total = 0
            running_loss = 0
            if module=='OPTICALCONV3D':
                total, running_loss = obj.train(self.trainset_loader, train_optical_loader)
            else:
                total, running_loss = obj.train(self.trainset_loader)
            print("|      {}     |   {}     |   {:.4f}  |               |".format(i, total, running_loss/total) )
            print("------------------------------------------------------")
            if  (i == EPOCHS) or testeverytrain:
                if module=='OPTICALCONV3D':
                    total, correct = obj.test(self.testset_loader, test_optical_loader)
                else:
                    total, correct = obj.test(self.testset_loader)
                print("|      {}     |    {}     |           | ({}) {:.1f} %   |".format(i, total, correct, 100 * correct/total))
                print("------------------------------------------------------")
            i += 1

    def __FFNN(self):
        loss = torch.nn.CrossEntropyLoss().to(self.device)
        out_features = 2                                        # only 2 classifier hit or miss
        in_features = self.trainset_loader.dataset[0][0].numel()
        network = FFNN.module.FFNN(in_features=in_features, out_features=out_features)
        if torch.cuda.device_count() > 1:                       # will use multiple gpu if available
            network = nn.DataParallel(network) 
        network.to(self.device)
        optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.4, nesterov=True)
        obj = FFNN.function.FFNNTraintest(self.device, network, loss, optimizer)
        return obj

    def __CNN3D(self):
        loss = torch.nn.CrossEntropyLoss().to(self.device)
        network = CNN3D.module.CNN3D(width=2*48, height=48, in_channels=3, out_features=2, drop_p=0.2, fc1out=256, fc2out=128, frames=100) # the shape of input will be Batch x Channel x Depth x Height x Width
        if torch.cuda.device_count() > 1:   #   will use multiple gpu if available
            network = nn.DataParallel(network) 
        network.to(self.device)
        optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.4, nesterov=True)
        obj = CNN3D.function.CNN3DTraintest(self.device, network, loss, optimizer)
        return obj

    def __CNN2DLSTM(self):
        loss = torch.nn.CrossEntropyLoss().to(self.device)
        encoder2Dnet = CNN2DLSTM.module.CNN2D(width=2*48, height=48, drop_p=0.2, fc1out=256, fc2out=128, frames=100) # Batch x Depth x Channel x Height x Width
        decoderltsm = CNN2DLSTM.module.LTSM()
        if torch.cuda.device_count() > 1:   #   will use multiple gpu if available
            encoder2Dnet = nn.DataParallel(encoder2Dnet)
            decoderltsm = nn.DataParallel(decoderltsm) 
        encoder2Dnet.to(self.device)
        decoderltsm.to(self.device)
        cnn_params = list(encoder2Dnet.parameters()) + list(decoderltsm.parameters())
        optimizer = torch.optim.SGD(cnn_params, lr=0.001, momentum=0.4, nesterov=True)
        #optimizer = torch.optim.Adam(cnn_params, lr=0.0001)
        obj = CNN2DLSTM.function.CNN2DLSTMTraintest(self.device, encoder2Dnet, decoderltsm, loss, optimizer)
        return obj

    def __OPTICALCONV3D(self, opticalpath):
        loss = torch.nn.CrossEntropyLoss().to(self.device)
        if opticalpath is None:
            raise RuntimeError('Please provide the path to opticalflow data')
        train_optical, test_optical = self.__get_opticalflow_view(self.trainset, self.testset, opticalpath)
        train_optical_loader = DataLoader(train_optical, shuffle=True)
        test_optical_loader = DataLoader(test_optical, shuffle=True)
        cnn3d = OPTICALCONV3D.module.CNN3D(width=2*48, height=48, in_channels=3, out_features=2, drop_p=0.2, fc1out=256, fc2out=128, frames=100) # the shape of input will be Batch x Channel x Depth x Height x Width
        opticalcnn3d = OPTICALCONV3D.module.CNN3D(width=2*48, height=48, in_channels=3, out_features=2, drop_p=0.2, fc1out=256, fc2out=128, frames=100) # the shape of input will be Batch x Channel x Depth x Height x Width
        twostream = OPTICALCONV3D.module.TwostreamConv3d()
        if torch.cuda.device_count() > 1:   #   will use multiple gpu if available
            cnn3d = nn.DataParallel(cnn3d)
            opticalcnn3d = nn.DataParallel(opticalcnn3d)
            twostream = nn.DataParallel(twostream) 
        cnn3d.to(self.device)
        opticalcnn3d.to(self.device)
        twostream.to(self.device)
        cnn_params = list(cnn3d.parameters()) + list(opticalcnn3d.parameters()) + list(twostream.parameters())
        optimizer = torch.optim.SGD(cnn_params, lr=0.001, momentum=0.4, nesterov=True)
        obj = OPTICALCONV3D.function.OPTICALCONV3DTraintest(self.device, cnn3d, opticalcnn3d, twostream, loss, optimizer)
        return obj, train_optical_loader, test_optical_loader

    def __get_opticalflow_view(self, trainset, testset, opticalpath):
        trainsetoptical = copy.deepcopy(trainset)
        trainsetoptical.samples = [[sample[0].replace(trainsetoptical.path, opticalpath), sample[1]] for sample in trainsetoptical.samples]
        trainsetoptical.path = opticalpath
        testsetoptical = copy.deepcopy(testset)
        testsetoptical.samples = [[sample[0].replace(testsetoptical.path, opticalpath), sample[1]] for sample in testsetoptical.samples]
        testsetoptical.path = opticalpath
        return trainsetoptical, testsetoptical
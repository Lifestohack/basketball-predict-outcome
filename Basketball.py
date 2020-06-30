#!/usr/bin/env python

import random
import torch
import torchvision
import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import copy
import torch.nn as nn
import serialize
import configparser
import os
import validation
from networks.FFNN import FFNN
from networks.CNN3D import CNN3D
from networks.CNN2DLTSM import CNN2DLTSM
from networks.TwoStream import TwoStream
from function import Traintest

class Basketball():
    def __init__(self, data, width=50, height=50, num_frames=100, split='training'):
        super().__init__()
        if not data or not num_frames:
            raise RuntimeError('Please pass parameter.')
        self.data = data
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.split = split
        self.channel = 3
        self.drop_p = 0.5
        self.out_features = 2                # only 2 classifier hit or miss
        self.lr = 0.001
        self.dense_flow = 'optics' 
        self.train_dense_loader = None
        self.test_dense_loader = None                                    
        
        dataset_training = Dataset.Basketball(self.data, split='training', num_frames = self.num_frames)
        trainset, testset = dataset_training.train_test_split(train_size=0.8)
        self.trainset_loader = DataLoader(trainset, shuffle=True)
        if self.split != 'validation':
            self.testset_loader = DataLoader(testset, shuffle=True)

        # dataset_validation = Dataset.Basketball(self.data, split='validation', num_frames = self.num_frames)
        # trainset, _ = dataset_validation.train_test_split(train_size=1)
        # self.trainset_loader = DataLoader(trainset, shuffle=True)

        if self.split == 'validation':
            trainset, _ = dataset_training.train_test_split(train_size=1)
            self.trainset_loader = DataLoader(trainset, shuffle=True)

        validation = dataset_training.getvalidation()
        self.validation_loader = DataLoader(validation, shuffle=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        # Config parser
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.config = config['DEFAULT']
        #a = config['trained_network']

    def run(self, module='FFNN', testeverytrain=True, EPOCHS=1):
        self.module = module
        if self.split == 'training':
            self.__runtraining(module, testeverytrain, EPOCHS)
        elif self.split == 'validation':
            self.__runvalidation(module, EPOCHS)

    def __module(self, module):
        obj = None
        if module=='FFNN':
            obj = self.__FFNN()
        elif module=='CNN3D':
            obj = self.__CNN3D()
        elif module=='CNN2DLSTM':
            obj = self.__CNN2DLSTM()
        elif module=='OPTICALCONV3D':
            obj = self.__OPTICALCONV3D()
        else:
            ValueError("Network {} doesnot exists.".format(module))
        return obj

    def __get_n_params(self, model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    def __FFNN(self):
        self.lr = 0.0001
        if self.num_frames == 55:
            self.lr = 0.0001
        elif self.num_frames == 30:
            self.lr = 0.0001
        loss = torch.nn.CrossEntropyLoss().to(self.device)
        in_features = self.width * self.height * self.channel * self.num_frames #self.trainset_loader.dataset[0][0].numel()
        network = FFNN(in_features=in_features, out_features=self.out_features, drop_p=self.drop_p)
        if torch.cuda.device_count() > 1:   # will use multiple gpu if available
            network = nn.DataParallel(network) 
        network.to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr, weight_decay=0.01)
        obj = Traintest(self.module, self.device, network, loss, optimizer)
        return obj, network

    def __CNN3D(self):
        self.lr = 0.0001
        if self.num_frames == 55:
            self.lr = 0.0001
        elif self.num_frames == 30:
            self.lr = 0.0001
        loss = torch.nn.CrossEntropyLoss().to(self.device)
        network = CNN3D(width=self.width, height=self.height, in_channels=self.channel, num_frames=self.num_frames, out_features=self.out_features, drop_p=self.drop_p) # the shape of input will be Batch x Channel x Depth x Height x Width
        if torch.cuda.device_count() > 1:                                   #   will use multiple gpu if available
            network = nn.DataParallel(network) 
        network.to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr, weight_decay=0.01)
        obj = Traintest(self.module, self.device, network, loss, optimizer)
        return obj, network

    def __CNN2DLSTM(self):
        self.lr = 0.0001
        if self.num_frames == 55:
            self.lr = 0.0001
        elif self.num_frames == 30:
            self.lr = 0.0001
        loss = torch.nn.CrossEntropyLoss().to(self.device)
        network = CNN2DLTSM(width=self.width, height=self.width, num_frames=self.num_frames, drop_p=self.drop_p,
                                                out_features=self.out_features, bidirectional=True)  
        if torch.cuda.device_count() > 1:   # will use multiple gpu if available
            network = nn.DataParallel(network)
        network.to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr, weight_decay=0.01)
        obj = Traintest(self.module, self.device, network, loss, optimizer)
        return obj, network

    def __OPTICALCONV3D(self):
        self.lr = 0.0001
        if self.num_frames == 55:
            self.lr = 0.0001
        elif self.num_frames == 30:
            self.lr = 0.0001
        
        self.trainset_loader.dataset.setOpticalflow(True)
        if self.split == 'training':
            self.testset_loader.dataset.setOpticalflow(True)
        self.validation_loader.dataset.setOpticalflow(True)
            
        loss = torch.nn.CrossEntropyLoss().to(self.device)
        if self.dense_flow is None:
            raise RuntimeError('Please provide the path to opticalflow data')
        network = TwoStream(width=self.width, height=self.height, in_channels=self.channel, 
                                            out_features=self.out_features, drop_p=self.drop_p, num_frames=self.num_frames) # the shape of input will be Batch x Channel x Depth x Height x Width
        if torch.cuda.device_count() > 1:                                                       # will use multiple gpu if available
            network = nn.DataParallel(network) 
        network.to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr, weight_decay=0.01)
        #obj = OPTICALCONV3D.function.OPTICALCONV3DTraintest(self.device, network, loss, optimizer)
        obj = Traintest(self.module, self.device, network, loss, optimizer)
        return obj, network

    def __get_opticalflow_view(self, trainset, testset, opticalpath):
        # deep copying the obj to new object for the optical flow #
        trainset_dense = copy.deepcopy(trainset)
        trainset_dense.path = self.dense_flow
        trainset_dense.samples = [sample.replace(self.data, self.dense_flow) for sample in trainset_dense.samples]
        testset_dense = copy.deepcopy(testset)
        testset_dense.path = opticalpath
        testset_dense.samples = [sample.replace(self.data, self.dense_flow) for sample in testset_dense.samples]
        return trainset_dense, testset_dense

    def __runtraining(self, module, testeverytrain, EPOCHS):
        print("Starting {} using the data in folder {}.".format(module, self.data ) )
        train_test, network = self.__module(module)
        print("Learning rate: {}".format(self.lr ) )
        num_parameter = self.__get_n_params(network)
        results = []
        results.append(['epochs','train','trainloss','test','correct','testloss'])
        for epoch in range(1, EPOCHS+1):
            result = []
            print('Epocs: ', epoch)
            total_train = len(self.trainset_loader.dataset)
            total_test = len(self.testset_loader.dataset)
            running_train_loss = train_test.train(self.trainset_loader)
            result.append(epoch)
            result.append(total_train)
            result.append(running_train_loss/total_train)
            print("\nTrain loss: {:.4f}".format(running_train_loss/total_train) )
            if  (epoch == EPOCHS) or testeverytrain:
                correct, running_test_loss = train_test.test(self.testset_loader)
                print("\nTest loss: {:.4f}, Prediction: ({}/{}) {:.1f} %".format(running_test_loss/total_test, correct, total_test, 100 * correct/total_test))
                result.append(total_test)
                result.append(correct)
                result.append(running_test_loss/total_test)
                results.append(result)
            print("-------------------------------------")
        prediction = train_test.predict(self.validation_loader)
        save_path = self.config['output']
        print("")
        print("Saving Validation results...")
        save_path_prediction = self.config['predictions']
        save_path_prediction_result = os.path.join(save_path, save_path_prediction)
        validationpath = serialize.exportcsv(prediction, modelclass=str(self.num_frames) + "_" + str(self.lr) + "_" + module, path=save_path_prediction_result)
        print("Done")
        print("Validating...")
        validation_result = validation.validate(validationpath)
        print("Done")
        #print("Saving network...")
        save_path = self.config['output']
        #save_path_network = self.config['trained_network']
        #save_path_trained_network = os.path.join(save_path, save_path_network)
        #module_saved_path = serialize.save_module(model=network, modelclass=module, path=save_path_trained_network)
        #for id,result in enumerate(results):
        #    if id==0:
        #        continue
        #    result.append(module_saved_path)
        # 'Epocs', 'total_train', 'running_loss_train', 'total_test', 'correct', 'test_loss', 'saved network'
        print("Saving Results...")
        save_path_results= self.config['results']
        save_path_results = os.path.join(save_path, save_path_results)
        serialize.save_results(results, modelclass=str(self.num_frames) + "_" + str(self.lr) + "_" +  str(validation_result) + "_" + module, path=save_path_results)
        #results = serialize.load_results('models\\results\\2020_05_15_12_08_22.csv') # to load the result
        print("Done")

    def __runvalidation(self, module, EPOCHS):
        print("Starting {} using the data in folder {}.".format(module, self.data ) )
        train_validate, network = self.__module(module)
        for epoch in range(1, EPOCHS+1):
            print('Epocs: ', epoch)
            train_validate.train(self.trainset_loader)
            print("")
        prediction = train_validate.predict(self.validation_loader)
        save_path = self.config['output']
        print("Saving Validation results...")
        save_path_prediction = self.config['predictions']
        save_path_prediction_result = os.path.join(save_path, save_path_prediction)
        validationpath = serialize.exportcsv(prediction, modelclass=str(self.num_frames) + "_" + str(self.lr) + "_" + module, path=save_path_prediction_result)
        print("Done")
        # print("Saving network...")
        # save_path_network = self.config['trained_network']
        # save_path_trained_network = os.path.join(save_path, save_path_network)
        # module_saved_path = serialize.save_module(model=network, modelclass=module, path=save_path_trained_network)
        # print("Done")
        print("Validating...")
        validation.validate(validationpath)
        print("Done")


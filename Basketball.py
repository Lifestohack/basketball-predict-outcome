#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Diwas Bhattarai
# Email: diwas@bhattarai.de

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
from networks.FFNN import FFNN
from networks.CNN3D import CNN3D
from networks.CNN2DLTSM import CNN2DLTSM
from networks.TwoStream import TwoStream
from function import Traintest
from function import TraintestPOSITIONLSTM
from networks.POSITIONFFNN import POSITIONFFNN
from networks.POSITIONLSTM import POSITIONLSTM
import cache
from networks.Networks import Networks
import validation

class Basketball():
    def __init__(self, width=48, height=48, split='training', trajectory=False):
        super().__init__()
        self.width = width
        self.height = height
        self.split = split
        self.num_frames = 100
        self.channel = 3
        self.drop_p = 0.5
        self.out_features = 2                # only 2 classifier hit or miss
        self.lr = 0.001
        self.data = None
        self.train_dense_loader = None
        self.test_dense_loader = None       
        self.trajectory = trajectory
        self.background = True
        self.optical = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        # Config parser
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.config = config['DEFAULT']
        #a = self.config['output']

    def run(self, num_frames, module=Networks.FFNN, testeverytrain=True, EPOCHS=1, lr=0.01, background=True, pretrained=False, pretrainedpath=None):
        if not num_frames:
            raise RuntimeError('Please pass parameter.')
        self.pretrained = pretrained
        self.pretrainedpath = pretrainedpath
        self.data = self.config['dataset']
        self.opticalpath = self.data
        self.background = background
        self.num_frames = num_frames
        self.lr = lr
        self.module = module
        if self.pretrained == True and self.pretrainedpath == None:
            self.pretrainedpath = self.getPretrainednetworkPath(self.module, self.trajectory, self.num_frames, self.background)
            if self.pretrainedpath == None:
                raise RuntimeError("No pretrained network found.")
        folder = "128x128"
        if module == Networks.FFNN:
            folder = "48x48"
        elif module == Networks.CNN3D or module == Networks.CNN3D or module == Networks.TWOSTREAM:
            folder = "128x128"
        if module == Networks.TWOSTREAM:
            folder_opticalpath = "128x128_optic"
        if self.trajectory == True:
            folder = "trajectory"
        
        if self.trajectory == False:
            if background:
                self.data = os.path.join(self.data, os.path.join("background", folder))   
                if module == Networks.TWOSTREAM:  
                    self.opticalpath = os.path.join(self.opticalpath, os.path.join("background", folder_opticalpath))    
            else:
                self.data = os.path.join(self.data, os.path.join("no_background", folder))   
                if module == Networks.TWOSTREAM: 
                    self.opticalpath = os.path.join(self.opticalpath, os.path.join("no_background", folder_opticalpath))
        elif self.trajectory == True:
            self.data = os.path.join(self.data, folder)         
        dataset_training = Dataset.Basketball(self.data, trajectory=self.trajectory)
        if self.split == "training":
            trainset, testset = dataset_training.train_test_split(train_size=0.8)
            if trainset == None and testset == None:
                raise FileNotFoundError("No training samples found.")
            self.trainset_loader = DataLoader(trainset, shuffle=True)
            self.testset_loader = DataLoader(testset, shuffle=True)
            self.trainset_loader.dataset.setFrames(self.num_frames)
            self.testset_loader.dataset.setFrames(self.num_frames)
        elif self.split == 'validation':
            if not self.pretrained:
                trainset, _ = dataset_training.train_test_split(train_size=1)
                if trainset == None:
                    raise FileNotFoundError("No training samples found.")
                self.trainset_loader = DataLoader(trainset, shuffle=True)
                self.trainset_loader.dataset.setFrames(self.num_frames)
        else:
            raise ValueError("Invalid argument: {}", self.split)
        validation = dataset_training.getvalidation()
        self.validation_loader = DataLoader(validation, shuffle=True)
        self.validation_loader.dataset.setFrames(self.num_frames)
        self.module = module
        if self.split == 'training':
            self.__runtraining(module, testeverytrain, EPOCHS)
        elif self.split == 'validation':
            self.__runvalidation(module, EPOCHS, pretrained, self.pretrainedpath)

    def __module(self, module):
        obj = None
        if module==Networks.FFNN:
            obj = self.__FFNN()
        elif module==Networks.CNN3D:
            obj = self.__CNN3D()
        elif module==Networks.CNN2DLSTM:
            obj = self.__CNN2DLSTM()
        elif module==Networks.TWOSTREAM:
            obj = self.__TWOSTREAM()
        elif module==Networks.POSITIONFFNN:
            obj = self.__POSITIONFFNN()
        elif module==Networks.POSITIONLSTM:
            obj = self.__POSITIONLSTM()
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
        loss = torch.nn.CrossEntropyLoss().to(self.device)
        network = CNN3D(width=self.width, height=self.height, in_channels=self.channel, num_frames=self.num_frames, out_features=self.out_features, drop_p=self.drop_p) # the shape of input will be Batch x Channel x Depth x Height x Width
        if torch.cuda.device_count() > 1:                                   #   will use multiple gpu if available
            network = nn.DataParallel(network) 
        network.to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr, weight_decay=0.01)
        obj = Traintest(self.module, self.device, network, loss, optimizer)
        return obj, network

    def __CNN2DLSTM(self):
        loss = torch.nn.CrossEntropyLoss().to(self.device)
        network = CNN2DLTSM(width=self.width, height=self.width, num_frames=self.num_frames, drop_p=self.drop_p,
                                                out_features=self.out_features)  
        if torch.cuda.device_count() > 1:   # will use multiple gpu if available
            network = nn.DataParallel(network)
        network.to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr, weight_decay=0.01)
        obj = Traintest(self.module, self.device, network, loss, optimizer)
        return obj, network       

    def __TWOSTREAM(self):
        self.optical = True
        if self.pretrained == False:
            self.trainset_loader.dataset.setOpticalflow(self.optical, self.opticalpath)
        if self.split == 'training':
            self.testset_loader.dataset.setOpticalflow(self.optical, self.opticalpath)
        self.validation_loader.dataset.setOpticalflow(self.optical, self.opticalpath)   
        loss = torch.nn.CrossEntropyLoss().to(self.device)
        if self.opticalpath is None:
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

    def __POSITIONFFNN(self):
        loss = torch.nn.CrossEntropyLoss().to(self.device)
        #in_features = self.width * self.height * self.channel * self.num_frames #self.trainset_loader.dataset[0][0].numel()
        #in_features = self.trainset_loader.dataset[0][0].numel()
        network = POSITIONFFNN(self.num_frames, self.out_features)
        if torch.cuda.device_count() > 1:   # will use multiple gpu if available
            network = nn.DataParallel(network) 
        network.to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr, weight_decay=0.01)
        obj = Traintest(self.module, self.device, network, loss, optimizer)
        return obj, network

    def __POSITIONLSTM(self):
        loss = torch.nn.CrossEntropyLoss().to(self.device)
        in_features = self.num_frames
        network = POSITIONLSTM(self.num_frames, in_features)  
        if torch.cuda.device_count() > 1:   # will use multiple gpu if available
            network = nn.DataParallel(network)
        network.to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr, weight_decay=0.01)
        obj = Traintest(self.module, self.device, network, loss, optimizer)
        return obj, network

    def __runtraining(self, module, testeverytrain, EPOCHS):
        print("Network: {} \nTotal Epocs: {} \nFrames: {}\nLearning rate: {}\nData: {}".format(module.name, EPOCHS, self.num_frames,self.lr, self.data))
        train_test, network = self.__module(module)
        if self.optical == True:
            print("Optical flow: {}".format(self.opticalpath)) 
        num_parameter = self.__get_n_params(network)
        results = []
        results.append(['epochs','train','trainloss','test','correct','testloss'])
        for epoch in range(1, EPOCHS+1):
            result = []
            print('+++++++++++++++++++++++++++++')
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
        save_path = self.config['output']
        print("")
        #print("Saving network...")
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
        backgroundpath = "trajectory"
        if self.trajectory == False:
            if self.background == True:
                backgroundpath = "background"
            elif self.background == False:
                backgroundpath = "no_background"
        save_path_results = os.path.join(save_path, backgroundpath, module.name, str(self.num_frames), save_path_results)
        saved_at = serialize.save_results(results, modelclass=str(EPOCHS) + "_" + str(self.lr) + "_" + module.name, path=save_path_results)
        print("Done!!! Saved at: ", saved_at)

    def __runvalidation(self, module, EPOCHS, pretrained, pretrainedpath):
        print("Network: {} \nTotal Epocs: {} \nFrames: {}\nLearning rate: {}\nData: {}".format(module.name, EPOCHS, self.num_frames,self.lr, self.data))
        train_validate, network = self.__module(module)
        if self.optical == True:
            print( "Optical flow: {}".format(self.opticalpath)) 
        if pretrained:
            network = serialize.load_module(network, pretrainedpath)
            prediction = train_validate.predict(self.validation_loader)
            #pre = validation.validate(prediction)
        else:
            total_train = len(self.trainset_loader.dataset)
            for epoch in range(1, EPOCHS+1):
                print("")
                print('Epocs: ', epoch)
                running_train_loss = train_validate.train(self.trainset_loader)
                print("")
                print("Training loss: ", running_train_loss/total_train)
                prediction = train_validate.predict(self.validation_loader)
                #pre = validation.validate(prediction)
        save_path = self.config['output']
        print("")
        print("Saving Validation results...")
        save_path_prediction = self.config['predictions']
        backgroundpath = "trajectory"
        if self.trajectory == False:
            if self.background == True:
                backgroundpath = "background"
            elif self.background == False:
                backgroundpath = "no_background"
        save_path_prediction_result = os.path.join(save_path, backgroundpath, module.name, str(self.num_frames), save_path_prediction)
        validationpath = serialize.exportcsv(prediction, modelclass=str(EPOCHS)+  "_" + str(self.lr) + "_" + module.name, path=save_path_prediction_result)
        print("Done!!! Saved at: ", validationpath)
        if not pretrained:
            print("Saving network...")
            save_path_network = self.config['trained_network']
            save_path_trained_network = os.path.join(save_path, backgroundpath, module.name, str(self.num_frames), save_path_network)
            saved_at = serialize.save_module(model=network, modelclass=str(EPOCHS)+ "_" + str(self.lr) + "_" + module.name, path=save_path_trained_network)
            print("Done!!! Saved at: ", saved_at)

    def destroycache(self):
        self.trainset_loader.dataset.destroycache()
        if self.split == 'training':
            self.testset_loader.dataset.destroycache()
        self.validation_loader.dataset.destroycache()
    
    def getPretrainednetworkPath(self, module, trajectory, num_frames, background):
        outputpath = self.config['output']
        files = []
        backgroundpath = "trajectory"
        if trajectory == False:
            if background == True:
                backgroundpath = "background"
            elif background == False:
                backgroundpath = "no_background"
        for r, d, f in os.walk(outputpath):
            for file in f:
                path = os.path.join(r, file)
                pathsplit = path.split(os.sep)
                if backgroundpath in pathsplit and str(num_frames) in pathsplit and module.name in pathsplit and "network" in pathsplit:
                    files.append(path)
        if len(files) > 1:
            print("More than one network found. Using the most recent trained network.")
        if len(files) < 1:
            raise RuntimeError("No trained network found.")
        return max(files, key=os.path.getctime)
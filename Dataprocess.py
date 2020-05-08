#!/usr/bin/env python3

import torch
import os
from PIL import Image
import torchvision
import numpy as np
import cv2
import denseflow

class Preprocess:
    def __init__(self, datasetpath=None, img_transformation=None, savepath='datatestcache', opticalflowpath=None, frompreprocessed=False, imagesize=(48,48)):
        if datasetpath is None:
            raise RuntimeError('No input path provided.')
        else:
            self.datasetpath = datasetpath
        self.img_transformation = img_transformation
        self.savepath = savepath
        self.opticalflowpath = opticalflowpath
        self.frompreprocessed = frompreprocessed
        self.imagesize = imagesize
        if not self.isprodataavailable(savepath):
            print('Data processing started......')
            self.preparedata(self.datasetpath)
            print('Data processing finished......')
        else:
            print('Data is already available.')
        

    def preparedata(self, data):
        listpaths = []
        datapaths = self.getsubdir(data)

    def getsubdir(self, path):
        filespath=[]
        subdir = os.listdir(path)
        for subpath in subdir:
            subsubpath = os.path.join(path, subpath)
            if os.path.isfile(subsubpath):
                filespath.append(subsubpath)
            else:
                subpathlist = self.getsubdir(subsubpath)
                if len(subpathlist) > 0:
                    img = self.removebackground(subpathlist)
                    self.saveimages(img, subpathlist)
                    if self.opticalflowpath is not None:
                        processfrom = self.datasetpath
                        if self.frompreprocessed:
                            subpathlist = [path.replace(self.datasetpath, self.savepath)  for path in subpathlist]
                            processfrom = self.savepath
                        #else:
                        #    toprocess = [path.replace(self.datasetpath, self.savepath)  for path in subpathlist]
                        denseflow.opticalflow(subpathlist, processfrom, self.opticalflowpath, self.imagesize)
        return filespath

    def removebackground(self, folder):
        video = []
        for file in folder:
            img = Image.open(file)
            if self.img_transformation is not None:
                img = self.img_transformation(img)           # should be always tensor
            if not isinstance(img, torch.Tensor):
                img = torchvision.transforms.ToTensor()(img) # should be converted to tensor even if img_transformation is None
            video.append(img)
        video = torch.stack(video)                  # stack the list of tensor to a big tensor
        #video = torch.stack(list(map(F.to_tensor, video))) if video is list of PIL image
        img = self.background_subtractor(method='median', tensor=video)
        return img

    def saveimages(self, tensor, path):
        savefilename = path[0].replace(self.datasetpath, self.savepath)
        savedir = os.path.dirname(savefilename)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        for index, img in enumerate(tensor):
            savefilename = path[index].replace(self.datasetpath, self.savepath)
            torchvision.utils.save_image(img, savefilename)

    def background_subtractor(self, method='median', tensor=None):
        if tensor is None:
            raise RuntimeError('No input tensor provided.')
        if method=='median':
            return self.meidan(tensor)
        else:
            raise ValueError()

    def meidan(self, tensor):
        background, _ = torch.median(tensor, dim=0, keepdim=True)
        ball_mask = (background - tensor).abs() > (50/255)
        ball = torch.where(ball_mask, tensor, torch.zeros_like(tensor))
        return ball

    def isprodataavailable(self, cache):
        cacheavailable = False
        if not os.path.isdir(cache):
            cacheavailable = False
        else:
            cacheavailable = True
        return cacheavailable


datapath = 'data'
afterprocessing = 'cache'
opticalflow = 'opticalflow'
imagesize = (48,48)
img_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(imagesize),
    torchvision.transforms.CenterCrop(imagesize),
    torchvision.transforms.ToTensor()
])

Preprocess(datapath, img_transformation=img_transform, savepath=afterprocessing, opticalflowpath=opticalflow, frompreprocessed=False, imagesize=imagesize)

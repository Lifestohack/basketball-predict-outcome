#!/usr/bin/env python3
import torch
from pathlib import Path
import os
from PIL import Image
import torchvision
from Dataprocess import Preprocess
import torchvision.transforms.functional as F
from torchvision.utils import save_image
import random
from torch.utils.data import DataLoader
import copy 

#   TODO
#   do i need to shuffel samples as it will be shuffel by dataloader??


# Classifier
# Hit = 1
# Miss = 2
class Basketball(torch.utils.data.Dataset):
    def __init__(self, path, split='training', num_frame=100, img_transform=torchvision.transforms.ToTensor(), dataprocess=None, cacheifavailable=True, savecache=False, combineview=False, cachepath='cache'):
        super().__init__()
        #split = training or validation
        #num_frame = 30, 50 or 100                                                                                                                             
        self.path = path
        self.split = split
        self.num_frame = num_frame
        self.img_transform = img_transform
        self.dataprocess = dataprocess
        self.cacheifavailable = cacheifavailable
        self.savecache = savecache
        self.combineview = combineview
        self.cachepath = cachepath
        self.length = 0
        self.hit = 'hit'
        self.miss = 'miss'
        self.samples = self._find_videos()
        self.samples = [i for i in self.samples]
        random.shuffle(self.samples)
        pass

    def __getitem__(self, index):
        path = self.samples[index]
        if self.miss == path[1]:
            label = 0
        elif self.hit == path[1]:
            label = 1
        cache = path[0].replace(self.path, self.cachepath)
        if self.cacheifavailable and self.iscacheavailable(cache):
            view = self.get_view(cache, cache=True)
            return view, label

        else:
            view1path = os.path.join(path[0], 'view1')
            view2path = os.path.join(path[0], 'view2')
            view1 = self.get_view(view1path)
            view2 = self.get_view(view2path)
            view = torch.cat((view1, view2), dim=3)
            view = view.transpose(2, 3) 
            if  self.savecache:
                if self.combineview:
                    viewpath =  path[0].replace(self.path, self.cachepath)
                    self.savecombinedcache(view, viewpath)
                else:
                    view1path = view1path.replace(self.path, self.cachepath)
                    view2path = view2path.replace(self.path, self.cachepath)
                    self.saveseperateview(view1, view2, view1path, view2path)
            return view, label

    def savecombinedcache(self, view, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        for index, _ in enumerate(view): 
            #imgpil1 = F.to_pil_image(view[index])
            savepath = os.path.join(path, str(index) + '.jpg')
            torchvision.utils.save_image(view[index], savepath)
            


    def saveseperateview(self, view1, view2, cache1, cache2):
        for index, _ in enumerate(view1): 
            imgpil1 = F.to_pil_image(view1[index])
            imgpil2 = F.to_pil_image(view2[index])
            if not os.path.isdir(cache1):
                os.makedirs(cache1)
            torchvision.utils.save_image(view1[index], os.path.join(cache1, str(index) + '.jpg'))
            if not os.path.isdir(cache2):
                os.makedirs(cache2)
            torchvision.utils.save_image(view2[index], os.path.join(cache2, str(index) + '.jpg'))

    def get_view(self, path, cache=False):
        frames_data = []
        frames = os.listdir(path)
        for idx, frame in enumerate(frames):
            img = Image.open(os.path.join(path, frame))
            if cache:
                #img = self.img_transform(img)
                #imgloaded = torch.load(os.path.join(path, frame))
                imgToTensor = torchvision.transforms.ToTensor()
                img = imgToTensor(img)
            else:
                if self.img_transform is not None:
                    img = self.img_transform(img)
                    #croparea = (18, 40, 260, 190)
                    #img = img.crop(croparea)
                    #img = torchvision.transforms.ToTensor()(img)
            frames_data.append(img)
            if idx == self.num_frame - 1:
                break
        video = torch.stack(frames_data)
        if not cache and self.dataprocess is not None: #test if not cache
            video = self.dataprocess(video)
            pass
        return video

    def __len__(self):
        return self.length

    def _find_videos(self):
        samples = []
        if self.split == 'training':
            hitsamples = self._getpath(self.hit)
            misssamples = self._getpath(self.miss)
            samples = hitsamples + misssamples
            pass
        elif self.split == 'validation':
            validationsamples = self._getpath()
            samples = validationsamples
        
        
        return samples

    def train_test_split(self, train_size = 0.8):
        hitsamples = [x for x in self.samples if x[1] == 'hit']
        hitnum = int(train_size * len(hitsamples))
        hittrainsamples = hitsamples[:hitnum] 
        hittestamples = hitsamples[hitnum:] 
        misssamples = [x for x in self.samples if x[1] == 'miss']
        missnum = int(train_size * len(misssamples))
        misstrainsamples = misssamples[:missnum] 
        misstestamples = misssamples[missnum:] 
        train = hittrainsamples + misstrainsamples
        test = hittestamples + misstestamples

        trainobj = copy.deepcopy(self)
        trainobj.samples = train
        trainobj.samples = [i for i in trainobj.samples]
        random.shuffle(trainobj.samples)
        trainobj.length = len(trainobj.samples)
        
        testobj = copy.deepcopy(self)
        testobj.samples = test
        testobj.samples = [i for i in testobj.samples]
        random.shuffle(testobj.samples)
        testobj.length = len(testobj.samples)
        
        return trainobj, testobj
        

    def _getpath(self, label=''):
        path = os.path.join(self.path, self.split, label)
        samples = []
        subdir = os.listdir(path)
        subdir = sorted(subdir, key=int)
        for item in subdir:
            sampleitem = []
            sampleitem.append(os.path.join(path, item))
            sampleitem.append(label)
            samples.append(sampleitem)
            self.length = self.length + 1
        return samples

    def iscacheavailable(self, cache1, cache2):
        cache = False
        if not os.path.isdir(cache1) or not os.path.isdir(cache2):
            cache = False
        else:
            cache = True
        return cache

    def iscacheavailable(self, cache):
        cacheavailable = False
        if not os.path.isdir(cache):
            cacheavailable = False
        else:
            cacheavailable = True
        return cacheavailable
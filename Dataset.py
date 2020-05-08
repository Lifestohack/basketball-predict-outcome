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
    def __init__(self, path, split='training', num_frame=100):
        super().__init__()
        #split = training or validation
        #num_frame = 30, 50 or 100                                                                                                                             
        self.path = path
        self.split = split
        self.num_frame = num_frame
        self.length = 0
        self.hit = 'hit'
        self.miss = 'miss'
        self.samples = self._find_videos()
        random.shuffle(self.samples)
        pass

    def __getitem__(self, index):
        path = self.samples[index]
        if path is None:
            print('No testdata on the folder ', index)
        if self.miss == path[1]:
            label = 0
        elif self.hit == path[1]:
            label = 1
        views = os.listdir(path[0])
        if len(views) == 2:
            view1path = os.path.join(path[0], views[0])
            view2path = os.path.join(path[0], views[0])
            view1 = self.get_view(view1path)
            view2 = self.get_view(view2path)
            view = torch.stack([view1, view2])
            return view, label
        else:
            view = self.get_view(path[0])
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

    def get_view(self, path):
        frames_data = []
        frames = os.listdir(path)
        for idx, frame in enumerate(frames):
            img = Image.open(os.path.join(path, frame))     # may be in future will save the processed images as tensor and not as image
            if not isinstance(img, torch.Tensor):
                img = torchvision.transforms.ToTensor()(img) # should be converted to tensor even if img_transformation is None
            frames_data.append(img)
            if idx == self.num_frame - 1:
                break
        video = torch.stack(frames_data)
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

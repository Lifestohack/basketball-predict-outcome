#!/usr/bin/env python3
import torch
from pathlib import Path
import os
from PIL import Image
import torchvision
from Dataprocess import Preprocess
import torchvision.transforms.functional as F


class Basketball(torch.utils.data.Dataset):
    def __init__(self, path, split='training', num_frame=100, img_transform=torchvision.transforms.ToTensor(), dataprocess=None):
        super().__init__()

        #split = training or validation
        #num_frame = 30, 50 or 100
                
        p1 = os.path.join(path, split)
        self.path = path

        self.split = split
        self.num_frame = num_frame
        self.img_transform = img_transform
        self.dataprocess = dataprocess
        self.length = 0
        self.hit = 'hit'
        self.miss = 'miss'
        self.samples = self._find_videos()
        pass
        
    def __getitem__(self, index):
        path = self.samples[index]
        view1path = os.path.join(path[0], 'view1')
        view2path = os.path.join(path[0], 'view2')
        
        if self.hit == path[1]:
            label = 1
        elif self.miss == path[1]:
            label = 0
        
        cache1 = view1path.replace('data', 'cache')
        cache2 = view2path.replace('data', 'cache')

        if self.iscacheavailable(cache1, cache2):
            return self.get_view(cache1, cache=True), self.get_view(cache2, cache=True), label
        else:
            view1 = self.get_view(view1path)
            view2 = self.get_view(view2path)
            self.cache(view1, view2, cache1, cache2)
            return view1, view2, label
    
    def cache(self, view1, view2, cache1, cache2):
        for index, _ in enumerate(view1): 
            imgpil1 = F.to_pil_image(view1[index])
            imgpil2 = F.to_pil_image(view2[index])
            if not os.path.isdir(cache1):
                os.makedirs(cache1)
            torch.save(imgpil1, os.path.join(cache1, str(index)))
            
            if not os.path.isdir(cache2):
                os.makedirs(cache2)
            torch.save(imgpil2, os.path.join(cache2, str(index)))

    def get_view(self, path, cache=False):
        frames_data = []
        subdir = os.listdir(path)
        if cache:
            frames = sorted(subdir, key=int)
        else:
            frames = sorted(subdir)

        for idx, frame in enumerate(frames):
            if cache:
                imgloaded = torch.load(os.path.join(path, frame))
                imgToTensor = torchvision.transforms.ToTensor()
                img = imgToTensor(imgloaded)
            else:
                img = Image.open(os.path.join(path, frame))
                if self.img_transform is not None:
                    img = self.img_transform(img)
            frames_data.append(img)
            if idx == self.num_frame - 1:
                break
        video = torch.stack(frames_data)
        if self.dataprocess is not None:
            video = self.dataprocess(video)
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

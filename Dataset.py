#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from pathlib import Path
import os
from PIL import Image
import torchvision
from torchvision.utils import save_image
import random
from torch.utils.data import DataLoader
import copy 
from cache import Cache
import time
import csv
import sys
import configparser
import numpy as np
import normalize
from sklearn.preprocessing import MinMaxScaler



# Classifier
# Hit = 1
# Miss = 2
class Basketball(torch.utils.data.Dataset):
    def __init__(self, path, trajectory=False):
        super().__init__()
        #split = training or validation
        #num_frames = 30, 50 or 100                                                                                                                             
        self.path = path
        self.length = 0
        self.curr_sample = None
        self.sample_num = 0
        self.optical_flow = False
        self.opticalpath = None
        self.trajectory = trajectory
        self.samples = self._find_videos()
        random.shuffle(self.samples)
        self.mean = [0,0,0]
        self.std = [0,0,0]
        # Config parser
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.config = config['DEFAULT']
        self.length = len(self.samples)     

    def setFrames(self, frames):
        self.num_frames = frames       

    def setOpticalflow(self, optical, path):
        self.optical_flow = optical
        self.opticalpath = path
    
    def useOpticalflow(self, optical):
        return self.optical_flow

    def __getitem__(self, index):
        cache = Cache()
        self.curr_sample = self.samples[index]
        if self.curr_sample is None:
            raise RuntimeError('No testdata on the folder ', index)
        item, isavaiable = cache.getcache(self.curr_sample)
        if isavaiable == True:
            num_frame_cache_available = None
            if self.trajectory == True:
                num_frame_cache_available = len(item[0][0])
            else:
                num_frame_cache_available = len(item[0][0]) if self.optical_flow == True else len(item[0])
            if num_frame_cache_available < self.num_frames:
                raise NotImplementedError("Don't Support running {} frames after running {} frame.".format(self.num_frames, num_frame_cache_available))
            view = torch.stack([item[0][0][:self.num_frames], item[0][1][:self.num_frames]]) if (self.optical_flow == True) else item[0][:self.num_frames]
            #view = torch.stack([item[0][0][:self.num_frames], item[0][1][:self.num_frames]]) if (self.optical_flow == True or self.trajectory == True) else item[0][:self.num_frames]
            return view, item[1]
        view = None
        view = self.__get_all_views(self.curr_sample)
        if self.optical_flow == True:
            if self.opticalpath is None:
                raise ValueError("No path for optical flow set.")
            curr_sample_optical_flow = self.curr_sample.replace(self.path, self.opticalpath)
            optical_flow = self.__get_all_views(curr_sample_optical_flow)
            view = torch.stack([view, optical_flow])
        cache.setcache(self.curr_sample, view, self.curr_sample)

        return view, self.curr_sample

    def __get_all_views(self, path):
        view = None
        views = os.listdir(path)
        if 'view1' in views:
            view1path = os.path.join(self.curr_sample, views[0])
            view2path = os.path.join(self.curr_sample, views[1])
            view1 = self.get_view(view1path)
            view2 = self.get_view(view2path)
            view = torch.stack([view1, view2])
        else:
            view = self.get_view(self.curr_sample)
            end = time.time()
        return view

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
            path_to_read = os.path.join(path, frame)
            if self.trajectory:
                with open(path_to_read, newline='') as csvfile:
                    frames_data = [list(map(float, row)) for row in csv.reader(csvfile)] 
                    frames_data = torch.FloatTensor(frames_data).unsqueeze(dim=0)
                    tensornormalize = torchvision.transforms.Compose([
                        torchvision.transforms.Normalize(mean=self.mean[0], std=self.std[0])
                    ])
                    frames_data = tensornormalize(frames_data).squeeze()
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    frames_data = scaler.fit_transform(frames_data)
                    frames_data = torch.FloatTensor(frames_data)
                    pass
            else:
                img = Image.open(path_to_read)     # may be in future will save the processed images as tensor and not as image
                tensornormalize = torchvision.transforms.Compose([
                     torchvision.transforms.ToTensor(),
                     torchvision.transforms.Normalize(mean=self.mean, std=self.std)
                ])
                img = tensornormalize(img) # should be converted to tensor
                frames_data.append(img)
                if idx == self.num_frames - 1:
                    break
        if self.trajectory:
            return frames_data
        else:
            video = torch.stack(frames_data)
            return video

    def __len__(self):
        return self.length

    def _find_videos(self):
        samples = []
        samples = self._getpaths(self.path)
        #samples = [x for x in samples if self.split in x]
        return samples

    def train_test_split(self, train_size = 0.8):
        samples = [x for x in self.samples if "hit" in x or "miss" in x]
        if len(samples) == 0:
            return None, None
        train_sample_num = int(train_size * len(samples))
        train_sample = samples[:train_sample_num] 
        test_sample = samples[train_sample_num:] 
        meanstd = normalize.calmeanstd(self.path, self.trajectory)
        trainobj = copy.deepcopy(self)
        trainobj.samples = train_sample
        random.shuffle(trainobj.samples)
        trainobj.length = len(trainobj.samples)
        mean = [0,0,0]
        std = [0,0,0]
        for item in trainobj.samples:
            if self.trajectory == True:
                mean += np.array((meanstd[item][0]))
                std += np.array((meanstd[item][1]))
            else:
                mean += np.array((meanstd[item][0], meanstd[item][1], meanstd[item][2]))
                std += np.array((meanstd[item][3], meanstd[item][4], meanstd[item][5]))
        trainobj.mean = mean/trainobj.length
        trainobj.std = std/trainobj.length
        if train_size == 1:
            return trainobj, None
        testobj = copy.deepcopy(self)
        testobj.samples = test_sample
        random.shuffle(testobj.samples)
        testobj.length = len(testobj.samples)
        mean = [0,0,0]
        std = [0,0,0]
        for item in testobj.samples:
            if self.trajectory == True:
                mean += np.array((meanstd[item][0]))
                std += np.array((meanstd[item][1]))
            else:
                mean += np.array((meanstd[item][0], meanstd[item][1], meanstd[item][2]))
                std += np.array((meanstd[item][3], meanstd[item][4], meanstd[item][5]))
        testobj.mean = mean/testobj.length
        testobj.std = std/testobj.length
        return trainobj, testobj

    def getvalidation(self):
        validationsamples = [x for x in self.samples if 'validation' in x]
        self.samples = validationsamples
        self.length = len(validationsamples)
        meanstd = normalize.calmeanstd(self.path, self.trajectory)
        mean = [0,0,0]
        std = [0,0,0]
        for item in self.samples:
            if self.trajectory == True:
                mean += np.array((meanstd[item][0]))
                std += np.array((meanstd[item][1]))
            else:
                mean += np.array((meanstd[item][0], meanstd[item][1], meanstd[item][2]))
                std += np.array((meanstd[item][3], meanstd[item][4], meanstd[item][5]))
        self.mean = mean/self.length
        self.std = std/self.length
        return self

    def _getpath(self, label=None):
        #obsolete
        hitormiss = ''
        if label is not None:
            hitormiss = label
        path = os.path.join(self.path, self.split, hitormiss)
        samples = []
        subdir = os.listdir(path)
        #subdir = sorted(subdir, key=int)
        for item in subdir:
            sampleitem = []
            sampleitem.append(os.path.join(path, item))
            sampleitem.append(hitormiss)
            samples.append(sampleitem)
            self.length = self.length + 1
        return samples

    def _getpaths(self, data_path):
        # get sample number folder without views
        go_deeper = True
        filespath = []
        subdir = os.listdir(data_path)
        for subpath in subdir:
            subsubpath = os.path.join(data_path, subpath)
            try: 
                int(subpath)
                go_deeper = False
                filespath.append(subsubpath)
            except ValueError:
                if go_deeper:
                    subsub_path_list = self._getpaths(subsubpath)   
                filespath += subsub_path_list                
        return filespath

    def isviewsavailable(self, cache1, cache2):
        cache = False
        if not os.path.isdir(cache1) or not os.path.isdir(cache2):
            cache = False
        else:
            cache = True
        return cache

    def isviewavailable(self, cache):
        cacheavailable = False
        if not os.path.isdir(cache):
            cacheavailable = False
        else:
            cacheavailable = True
        return cacheavailable

    def destroycache(self):
        cache = Cache()
        cache.destroy()
        del cache
#!/usr/bin/env python3

import torch
import os
from PIL import Image
import torchvision
import numpy as np
import cv2 as cv
import denseflow
from skimage import io, transform
import numpy
import time
import shutil
from backgroundAveraging import BackGroundSubtractor


class Preprocess:
    def __init__(self, 
                dataset_path=None, 
                save_path=None, 
                of_save_path=None, 
                pp_opt_flow=False, 
                resize=None):

        if dataset_path is None:
            raise RuntimeError('No input path provided.')
        else:
            self.dataset_path = dataset_path

        if save_path is None and of_save_path is None:
            raise RuntimeError('Please provide at least one path.')

        self.save_path = save_path
        self.of_save_path = of_save_path
        self.pp_opt_flow = pp_opt_flow
        self.resize = resize
        
    def prepare_data(self, data_path):
        if data_path is None:
            data_path = self.dataset_path
        filespath = []
        subdir = os.listdir(data_path)
        for subpath in subdir:
            subsubpath = os.path.join(data_path, subpath)
            if os.path.isfile(subsubpath):
                filespath.append(subsubpath)
            else:
                subsub_path_list = self.prepare_data(subsubpath)
                if len(subsub_path_list) > 0:
                    self.__data_process(subsub_path_list)
        return filespath

    def __data_process(self, subsubpath):
        processdir = os.path.dirname(subsubpath[0])
        print('Data processing at', processdir)
        org_video = self.get_video(subsubpath)
        video = self.re_size(org_video)
        if self.save_path and self.of_save_path:
            transformed_video = self.img_transform(video, subsubpath, save=True)
            if self.pp_opt_flow:
                self.dense_transform(transformed_video, subsubpath,  save=True)
            else:
                self.dense_transform(org_video, subsubpath, save=True)
        elif self.save_path:
            self.img_transform(video, subsubpath, save=True)
        elif self.of_save_path:
            if self.pp_opt_flow:
                transformed_video = self.img_transform(video, subsubpath, save=False)
                self.dense_transform(transformed_video,  save=True)
            else:
                self.dense_transform(video, subsubpath, save=True)
        else:
            raise RuntimeError('Weird!!! I am not suppose to be here. Something went wrong during __data_process()')


    def img_transform(self, video, subsubpath, save=False):
        no_background_video = self.remove_background(video)
        if save:
            if subsubpath is None:
               raise RuntimeError('Please provide the original frames file paths list.')
            save_here = [path.replace(self.dataset_path, self.save_path)  for path in subsubpath]
            self.save_images(no_background_video, save_here)
        return no_background_video

    def dense_transform(self, video, subsubpath,  save=False):
        dense_video = denseflow.get_optical_flow(video)
        dense_video = self.re_size(dense_video)
        if save:
            if subsubpath is None:
                raise RuntimeError('Please provide the original frames file paths list.')
            optical_save_path = [path.replace(self.dataset_path, self.of_save_path)  for path in subsubpath]
            self.save_images(dense_video, optical_save_path)
        return dense_video

    def remove_background(self, videos):
        img = self.__background_subtractor('custom', videos)
        return img

    def get_video(self, folder):
        video = []
        for imagepath in folder:
            #img = Image.open(imagepath)
            #img = numpy.array(img)
            img = cv.imread(imagepath)
            video.append(img)
        return video
    
    def re_size(self, video_array):
        resized_video = []
        if self.resize: 
            for frame in video_array:                        # resize takes place here
                img = cv.resize(frame, self.resize)
                resized_video.append(img)
            return resized_video
        else:
            return video_array

    def save_images(self, video, path):
        savedir = os.path.dirname(path[0])
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        for index, img in enumerate(video):
            #torchvision.utils.save_image(img, path[index]) #save tensor
            cv.imwrite(path[index], img)
        print('Data saved at', savedir)

    def __destroy(self, dir):
        if os.path.exists(dir):
            print('Cache will be deleted in 10 seconds. Cancel now if it was a mistake.')
            time.sleep(10)
            shutil.rmtree(dir)

    def __background_subtractor(self, method, videos):
        if videos is None:
            raise RuntimeError('No input tensor provided.')
        if method=='custom':
            return self.__custom_background_remover(videos)
        else:
            raise ValueError()

    def __denoise(self, frame):
        frame = cv.medianBlur(frame,5)
        frame = cv.GaussianBlur(frame,(5,5),0)
        return frame

    def __meidan(self, frames):
        background, _ = torch.median(frames, dim=0, keepdim=True)
        ball_mask = (background - frames).abs() > (50/255)
        ball = torch.where(ball_mask, frames, torch.zeros_like(frames))
        return ball

    def __custom_background_remover(self, videos):
        videos = iter(videos)
        prev = next(videos, None)
        backSubtractor = BackGroundSubtractor(0.4, self.__denoise(prev))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        without_background = []
        for frame in videos:
            frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
            foreGround = backSubtractor.getForeground(self.__denoise(frame))
            ret, mask = cv.threshold(foreGround, 15, 255, cv.THRESH_BINARY)
            #cv.imshow('frame',mask)
            #cv.waitKey(0)
            without_background.append(mask)
        return without_background


    def __custom_background_remover_(self, videos):
        fgbg = cv.createBackgroundSubtractorMOG2()
        without_background = []
        for frame in videos:
            fgmask = fgbg.apply(frame)
            fgmask = self.__denoise(fgmask)
            #cv.imshow('frame',fgmask)
            #cv.waitKey(0)
            without_background.append(fgmask)
        return without_background

    def __isprodataavailable(self, cache):
        cacheavailable = False
        if not os.path.isdir(cache):
            cacheavailable = False
        else:
            cacheavailable = True
        return cacheavailable

# dataset_path = 'data'
# save_path = 'cache1'
# of_save_path = 'optics'
# pp_opt_flow = False
# resize = (224, 224)

# process = Preprocess(dataset_path=dataset_path, 
#                 save_path=save_path, 
#                 of_save_path=of_save_path, 
#                 pp_opt_flow=pp_opt_flow, 
#                 resize=resize)

# process.prepare_data(None)
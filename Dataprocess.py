#!/usr/bin/env python3

import torch
import os
from PIL import Image
import torchvision
import numpy as np
import cv2 as cv
import denseflow
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
        
    def prepare_data(self):
        return_save_path = False
        if self.save_path is not None:
            if self.__is_pro_data_aavailable(self.save_path):
                print('Data are already available at ', self.save_path)
                return_save_path = True
        else:
            return_save_path = True
        return_op_save_path = False
        if self.of_save_path is not None:      
            if self.__is_pro_data_aavailable(self.of_save_path):
                print('Optical flow data are already available at ', self.of_save_path)
                return_op_save_path = True
        else:
            return_op_save_path = True
        if not return_save_path or not return_op_save_path:
            self.__recursive_prepare_data(self.dataset_path)

    def __recursive_prepare_data(self, data_path):
        filespath = []
        subdir = os.listdir(data_path)
        for subpath in subdir:
            subsubpath = os.path.join(data_path, subpath)
            if os.path.isfile(subsubpath):
                filespath.append(subsubpath)
            else:
                subsub_path_list = self.__recursive_prepare_data(subsubpath)
                if len(subsub_path_list) > 0:
                    self.data_process(subsub_path_list)
        return filespath

    def data_process(self, subsubpath):
        processdir = os.path.dirname(subsubpath[0])
        print('Data processing at', processdir)
        org_video = self.get_video(subsubpath)
        #video = self.re_size(org_video, (360, 480)) # H x W
        if self.save_path and self.of_save_path:
            transformed_video = self.img_transform(org_video, subsubpath, save=True)    # returns without resize
            if self.pp_opt_flow:
                self.dense_transform(transformed_video, subsubpath,  save=True)
            else:
                self.dense_transform(org_video, subsubpath, save=True)
        elif self.save_path:
            self.img_transform(org_video, subsubpath, save=True)
        elif self.of_save_path:
            if self.pp_opt_flow:
                transformed_video = self.img_transform(org_video, subsubpath, save=False)
                self.dense_transform(transformed_video, subsubpath,  save=True)
                pass
            else:
                self.dense_transform(org_video, subsubpath, save=True)
        else:
            raise RuntimeError('Weird!!! I am not suppose to be here. Something went wrong during data_process()')

    def img_transform(self, video, subsubpath, save=False):
        no_background_video = self.remove_background(video)
        resize_no_background_video = self.re_size(no_background_video, self.resize)
        if save:
            if subsubpath is None:
               raise RuntimeError('Please provide the original frames file paths list.')
            save_here = [path.replace(self.dataset_path, self.save_path)  for path in subsubpath]
            self.save_images(resize_no_background_video, save_here)
        return no_background_video

    def dense_transform(self, video, subsubpath,  save=False):
        # video should be array and is array before it is saved
        dense_video = denseflow.get_optical_flow(video)
        dense_video = self.re_size(dense_video, self.resize)
        if save:
            if subsubpath is None:
                raise RuntimeError('Please provide the original frames file paths list.')
            optical_save_path = [path.replace(self.dataset_path, self.of_save_path)  for path in subsubpath]
            self.save_images(dense_video, optical_save_path)
        return dense_video

    def remove_background(self, videos):
        img = self.__background_subtractor('median', videos)
        return img

    def get_video(self, folder):
        video = []
        for imagepath in folder:
            img = Image.open(imagepath)
            video.append(img)
        video = np.stack(video)
        return video
    
    def re_size(self, video, re_size_scale):
        #takes array and returns array
        resized_video = []
        if re_size_scale:
            org_height =  video[0].shape[0]
            org_width =  video[0].shape[1]
            ratio = 1
            view1 = True
            if org_height > org_width:
                view1 = False
                ratio = org_height / org_width
            else:
                ratio = org_width / org_height

            for frame in video:
                #resampling using pixel area relation. It may be a preferred method for image decimation, 
                # as it gives moire'-free results. But when the image is zoomed, 
                # it is similar to the INTER_NEAREST method
                # output = cv2.resize(src, dsize, interpolation = cv2.INTER_AREA)
                #INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4  
                # re_size_scale -> H x W         
                height = re_size_scale[0]
                width = int(ratio * height)
                if view1:
                    frame_resize = cv.resize(frame, (height, width))    # uses cv.INTER_AREA interpolation. 
                else: 
                    frame_resize = cv.resize(frame, (width, height))    # uses cv.INTER_AREA interpolation. 



                frame_crop = self.crop_center(frame_resize, re_size_scale[0], re_size_scale[1])
                resized_video.append(frame_crop)
            resized_video = np.stack(resized_video) 
            return resized_video
        else:
            return video

    def crop_center(self, img, cropx, cropy):
        y,x,z = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx]

    def save_images(self, video, path):
        # video can be tensor or array
        savedir = os.path.dirname(path[0])
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        if isinstance(video, torch.Tensor):
            for index, img in enumerate(video):
                torchvision.utils.save_image(img, path[index]) #save tensor
        else:
            for index, img in enumerate(video):
                cv.imwrite(path[index], img) # if shape is h x w x c x frames -> video[...,index]
        print('Data saved at', savedir)

    def re_size_PIL(self, video):
        # video is list of PIL image and returns list of PIL image
        transform =   torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.resize),
            torchvision.transforms.CenterCrop(self.resize)
        ])
        resized_video = []
        if self.resize:
            for frame in video:
                frame_resize = transform(frame)
                resized_video.append(frame_resize)
            return resized_video
        else:
            return video

    def __destroy(self, dir):
        if os.path.exists(dir):
            print('Cache will be deleted in 10 seconds. Cancel now if it was a mistake.')
            time.sleep(10)
            shutil.rmtree(dir)

    def __background_subtractor(self, method, videos):
        if videos is None:
            raise RuntimeError('No input tensor provided.')
        if method=='median':
            return self.__meidan_array(videos, 50)
        else:
            raise ValueError()

    def __meidan_tensor(self, frames, rate=30):
        zeross = torch.zeros_like(frames)                         # 100 x 3 x 48 x 48
        background, _ = torch.median(frames, dim=0, keepdim=True) # 1 x 3 x 48 x 48
        ball_mask = (background - frames).abs() > (rate/255)        # 100 x 3 x 48 x 48
        ball = torch.where(ball_mask, frames, zeross)             # 100 x 3 x 48 x 48
        return ball

    def __meidan_array(self, video, rate=30):                                       
        background = np.median(video, axis=0, keepdims=True).astype(np.uint8)
        foreground = np.absolute((background - video))    
        foreground[foreground > (255 - rate)] = 0
        foreground[foreground < (0 + rate)] = 0          
        #Image.fromarray(foreground[0]).save('result.png')
        return foreground

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

    def __is_pro_data_aavailable(self, cache):
        cacheavailable = False
        if not os.path.isdir(cache):
            cacheavailable = False
        else:
            cacheavailable = True
        return cacheavailable
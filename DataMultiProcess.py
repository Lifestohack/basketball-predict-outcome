#!/usr/bin/env python
# -*- coding: utf-8 -*-

from multiprocessing import Pool
import os
import cv2 as cv
import numpy as np
import denseflow
from PIL import Image
import time

class DataMultiProcess():
    def __init__(self, dataset_path, save_path, create_dense, resize, removebackground=True):
        super().__init__()
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.create_dense = create_dense
        self.resize = resize
        self.removebackground = removebackground
        self.crop_width = 768
        self.crop_height = 384

    def get_views(self, data_path):
        go_deeper = True
        filespath = []
        subdir = os.listdir(data_path)
        for subpath in subdir:
            subsubpath = os.path.join(data_path, subpath)
            if 'view' in subsubpath:
                go_deeper = False
                filespath.append(subsubpath)
            else:
                if go_deeper:
                    subsub_path_list = self.get_views(subsubpath)   
                filespath += subsub_path_list                
        return filespath

    def get_sample_folder_number(self, data_path):
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
                    subsub_path_list = self.get_sample_folder_number(subsubpath)   
                filespath += subsub_path_list                
        return filespath

    def crop_resize_concatenate(self, sample):
        # takes folder with sampel number (not view folder) and concatenates those views and saves them
        views = os.listdir(sample)
        views = [os.path.join(sample, view) for view in views]
        videos = []
        frame_names = []
        for view in views:
            video = self.crop_rotate_view(view, True)       # rotating only view2
            videos.append(video)
            frame_names = os.listdir(view)  #it will be later after the loop ends
        frame_names = [os.path.join(sample, frame) for frame in frame_names]
        frame_names = [framessortedname.replace(self.dataset_path, self.save_path) for framessortedname in frame_names]
        combo_frames = np.concatenate(videos, axis=1)
        i = 0
        for combo_frame in combo_frames:
            folder = os.path.dirname(frame_names[i])
            if not os.path.isdir(folder):
                os.makedirs(folder)
            cv.imwrite(frame_names[i], combo_frame)
            i+=1
        print('Saved:  ', folder)

    def crop_rotate_view(self, view, rotate):
        # Takes view folder and returns all the cropped images as array
        # Takes argument rotate as boolean to check if you want to rotate it 
        # rotates only view2 
        frames = os.listdir(view)
        frames = [os.path.join(view, frame) for frame in frames]
        width = self.crop_width
        height = self.crop_height
        view1 = True
        if 'view2' in frames[0]:
            view1 = False
            width = self.crop_height
            height = self.crop_width
        video = []
        for frame in frames:
            img = cv.imread(frame)
            img = self._crop_image(img, width, height, view1)
            img = self.resize_rotate(img, view1)
            video.append(img)
        return video    

    def resize_rotate(self, img, view1):
        # takes img and checks if it is view1 or view2 and returns resized and rotated image
        #h = int(self.crop_height/3)
        #w = int(self.crop_width/3)
        h=self.resize[0]
        w=self.resize[1]
        if view1:
            img = cv.resize(img, (w,h))
        else:
            img = cv.resize(img, (h,w))
            img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        #cv.imshow("cropped", img)
        #cv.waitKey(0)
        return img

    def _crop_image(self, img, cropx, cropy, view1):
        y,x,z = img.shape
        xx = 40
        yy = 60
        if not view1:
            xx = 100
            yy = - 30
        startx = x//2-(cropx//2) - xx
        starty = y//2-(cropy//2) - yy
        crop_img = img[starty:starty+cropy,startx:startx+cropx]
        # if not view1:
        #cv.imshow("cropped", crop_img)
        #cv.waitKey(0)
        return crop_img

    def _resize(self, sample):
        frames = os.listdir(sample)
        frames = [os.path.join(sample, view) for view in frames]
        for frame in frames:
            img = cv.imread(frame)
            img = cv.resize(img, self.resize)
            frame_name = frame.replace(self.dataset_path, self.save_path) 
            folder = os.path.dirname(frame_name)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            cv.imwrite(frame_name, img)
        print('Saved: ', frame_name)

    def rotate(self, sample, path, rotate):
        rotate_path = path
        if path is None:
            rotate_path =  self.save_path
        frames = os.listdir(sample)
        frames = [os.path.join(sample, frame) for frame in frames]
        for frame in frames:
            img = cv.imread(frame)
            img = cv.rotate(img, rotate)
            frame = frame.replace(self.save_path, rotate_path)   # i am drunk and I dont know why I put this block in if clause
            folder = os.path.dirname(frame)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            cv.imwrite(frame, img)
        print('Saved:  ', folder)

    def pipeline(self, sample):
        views, frames = self.__get_nobackground_cropped_views(sample, True)
        # Concatenation of two views starts here #
        combo_video = np.concatenate(views, axis=1)
        # Concatenation of two views starts here #
        folder = "no_background"
        if self.removebackground == False:
            folder = "background"
        # Saving videos starts here #
        #frames = [path.replace(self.dataset_path, self.save_path) for path in frames[0]] # remove data_set path to save_path
        frames_to_Save = []
        for path in frames[0]:
            pathsplit = path.split(os.sep)
            pathsplit[0] = self.save_path
            to_remove = pathsplit[-2]
            pathsplit.remove(to_remove)
            pathsplit.insert(1, folder)
            sizepath = str(resize[1]) + "x" + str(resize[1])
            if self.create_dense == True:
                sizepath = sizepath + "_optic"
            pathsplit.insert(2, sizepath)
            pathsplit.insert(3, "basketball") 
            frames_to_Save.append(os.path.join(*pathsplit))
        self.save_video(combo_video, frames_to_Save)

        if "validation" not in frames_to_Save[0].lower():
            # Rotate starts here #
            ROTATE_90_COUNTERCLOCKWISE = self.rotate_video(combo_video, cv.ROTATE_90_COUNTERCLOCKWISE)
            ROTATE_90_CLOCKWISE = self.rotate_video(combo_video, cv.ROTATE_90_CLOCKWISE)
            ROTATE_180 = self.rotate_video(combo_video, cv.ROTATE_180)
            # Rotate ends here #
            frames_to_Save_rotate_90_counterclockwise = []
            for path in frames_to_Save:
                pathsplit = path.split(os.sep)
                pathsplit[3] = pathsplit[3] + "_rotate_90_counterclockwise"
                frames_to_Save_rotate_90_counterclockwise.append(os.path.join(*pathsplit))
            self.save_video(ROTATE_90_COUNTERCLOCKWISE, frames_to_Save_rotate_90_counterclockwise)
            frames_to_Save_rotate_90_clockwise = []
            for path in frames_to_Save:
                pathsplit = path.split(os.sep)
                pathsplit[3] = pathsplit[3] + "_rotate_90_clockwise"
                frames_to_Save_rotate_90_clockwise.append(os.path.join(*pathsplit))
            self.save_video(ROTATE_90_CLOCKWISE, frames_to_Save_rotate_90_clockwise)
            frames_to_Save_rotate_180 = []
            for path in frames_to_Save:
                pathsplit = path.split(os.sep)
                pathsplit[3] = pathsplit[3] + "_rotate_180"
                frames_to_Save_rotate_180.append(os.path.join(*pathsplit))
            self.save_video(ROTATE_180, frames_to_Save_rotate_180)

    def pipeline_no_background_crop(self, sample):
        views, frames = self.__get_nobackground_cropped_views(sample, False)
        folder = "matlab"
        # Saving videos starts here #
        for view, frame in zip(views, frames):
            new_frame_names = []
            for f in frame:
                pathsplit = f.split(os.sep)
                pathsplit.insert(1, folder)
                new_frame_names.append((os.path.join(*pathsplit)))
            replaced_new_frame_names = [path.replace(self.dataset_path, self.save_path) for path in new_frame_names] # remove data_set path to save_path
            self.save_video(view, replaced_new_frame_names)
        # Saving videos Ends here #

    def __get_nobackground_cropped_views(self, sample, rotateresize):
        views = os.listdir(sample)
        views = [os.path.join(sample, view) for view in views]
        bothviews = []
        bothviewspath = []
        for view in views:
            frames = os.listdir(view)
            frames = [os.path.join(view, frame) for frame in frames]
            video = []
            frames.sort()
            # Read images from view starts here #
            for frame in frames:
                img = cv.imread(frame)
                video.append(img)
            # Read images from view ends here #

            removed_background_video = video
            if self.removebackground:
                # remove the background of the previously read view starts here #
                removed_background_video = self.remove_background(video)
                # remove the background of the previously read view starts here #

            view1 = False
            width = self.crop_width                 # view1 is horizontal video
            height = self.crop_height
            if "view1" in view:
                view1 = True
            else:
                view1 = False
                width = self.crop_height            # view2 is vertical video so width and height changes
                height = self.crop_width
            
            # Dense Video starts here #
            if self.create_dense == True:
                removed_background_video = self.dense_video(removed_background_video)
            # Dense Video end here #

            # Crop resize and rotate if view 2 starts here #            
            cropped_resize_rotate_removed_background_video = []
            for frame in removed_background_video:
                frame = self._crop_image(frame, width, height, view1)
                if rotateresize:
                    frame = self.resize_rotate(frame, view1)        # rotate only if view 2
                cropped_resize_rotate_removed_background_video.append(frame)
            # Crop resize and rotate if view 2 ends here #
            bothviews.append(cropped_resize_rotate_removed_background_video)
            bothviewspath.append(frames)
        return bothviews, bothviewspath

    def remove_background(self, video):  
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)) #10,10
        backSub = cv.createBackgroundSubtractorMOG2()
        #backSub = cv.createBackgroundSubtractorKNN()
        removed_background = []
        for i, frame in enumerate(video):
            mask = backSub.apply(frame)
            if i == 0:
                continue
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            #gmask = cv.erode(gmask,kernel,iterations = 1)
            fgMask = mask[:, :, None] * np.ones(3, dtype=int)[None, None, :]
            foreground = frame * (fgMask!=0)
            removed_background.append(foreground)
            #cv.imshow("cropped", foreground)
            #keyboard = cv.waitKey(30)
            #if keyboard == 'q' or keyboard == 27:
            #    break
        return removed_background
    
    def dense_video(self, video):
        return denseflow.get_optical_flow(video)

    def rotate_video(self, video, rotate):
        frames = []
        for img in video:
            img = cv.rotate(img, rotate)
            frames.append(img)
        return frames

    def save_video(self, video, frames):
        folder = os.path.dirname(frames[0])
        if not os.path.isdir(folder):
            os.makedirs(folder)
        for frame, path in zip(video, frames):
            cv.imwrite(path, frame)
        print("Saved:", folder)

    def start(self):
        if __name__ == '__main__':
            start_time = time.time()
            if self.create_dense == True:
                print("Creating dense dataset...")
            else:
                print("Creating dataset...")
            # Dataset processing
            views = self.get_sample_folder_number(self.dataset_path)
            #self.pipeline_no_background_crop(views[0])
            try:
                # use crop_resize_concatenate for resizing concatenating and saving
                # use rotate to rotate images
                pool = Pool()                  # Create a multiprocessing Pool. Pass number of cores to use as argument
                if self.resize == None:
                    pool.map(self.pipeline_no_background_crop, views)    # process views iterable with pool 
                else:
                    pool.map(self.pipeline, views)    # process views iterable with pool
            finally:                            # To make sure processes are closed in the end, even if errors happen
                pool.close()
                pool.join()
            end_time = time.time()
            print("Total time required: {} seconds".format(end_time - start_time))

# *************<IMPORTANT>*********************
# When running this file make sure
# only one call to DataMultiProcess is made.
# Comment out other call to DataMultiProcess

# Process-based parallelism
# This file will use all the logical cores on 
# your computer. If you don't want that then 
# pass number of cores you want it to use when
# calling the Pool(). Using all cores will make 
# dataprocessing faster.

# Original data path and save path after preprocessing
original_data_path = "orgdata"
save_path = "dataset"
if not os.path.exists(original_data_path):
    raise FileNotFoundError(original_data_path + " not found.")

# Make sure the original dataset has path structure as follows:
# Case sensative and training and validation folder are available

# -originaldataset
#       - training
#       - validation

# At the end of data processing following path structure will be formed.
# - dataset
#       - background
#           - 48x48
#               - basketball
#                   - training
#                   - validation
#               - basketball_rotate_90_clockwise
#                   - training
#               - basketball_rotate_90_counterclockwise
#                   - training
#               - basketball_rotate_180
#                   - training
#           - 128x128
#           - 128x128_optic
#
# there will also be folder with same structure with no_background
# - dataset
#       - no_background
#           ...
# *************</IMPORTANT>*********************


# Create dataset for FFNN.
# Each images will be resized
# After resize two views will be concatenated
# Resulting concatenated images size will be of size 48x48.
# deactive dense creation
# Two dataset will be created 
# With background and without background

#create_dense = False
#resize = [24, 48]
#DataMultiProcess(original_data_path, save_path, create_dense, resize, removebackground=True).start()
#DataMultiProcess(original_data_path, save_path, create_dense, resize, removebackground=False).start()


# Create dataset for CNN3D, CNN2DLSTM, TWOSTREAM
# Two dataset will be created 
# With background and without background

#create_dense = False
#resize = [64, 128]
#DataMultiProcess(original_data_path, save_path, create_dense, resize, removebackground=True).start()
#DataMultiProcess(original_data_path, save_path, create_dense, resize, removebackground=False).start()


# Create dense dataset for TWOSTREAM
# active dense creation
# Two dataset will be created 
# Dense flow images will be created after removing background
# Dense flow images will be created without removing background

#create_dense = True
#resize = [64, 128]
#DataMultiProcess(original_data_path, save_path, create_dense, resize, removebackground=True).start()
#DataMultiProcess(original_data_path, save_path, create_dense, resize, removebackground=False).start()


# Create dataset for matlab input
# This dataset will be used to extract the position and radius of the basketball.

#create_dense = False
#resize = None
#DataMultiProcess(original_data_path, save_path, create_dense, resize, removebackground=True).start()
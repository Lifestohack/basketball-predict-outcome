from multiprocessing import Pool
import Dataprocess
import os
import cv2 as cv
import numpy as np

class DataMultiProcess():
    def __init__(self, dataset_path, save_path, of_save_path, pp_opt_flow, resize):
        super().__init__()
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.of_save_path = of_save_path
        self.pp_opt_flow = pp_opt_flow
        self.resize = resize
        self.crop_width = 768
        self.crop_height = 384
        self.process = Dataprocess.Preprocess(dataset_path=self.dataset_path, 
                            save_path=self.save_path, 
                            of_save_path=self.of_save_path, 
                            pp_opt_flow=self.pp_opt_flow, 
                            resize=self.resize)

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

    def process_view(self, view):
        # starts the process for class Dataprocess and uses Multi process
        frames_path = os.listdir(view)
        frames_path = [os.path.join(view, frame) for frame in frames_path ]
        self.process.data_process(frames_path)

    def crop_resize_concatenate(self, sample):
        # takes folder with sampel number (not view folder) and concatenates those views and saves them
        views = os.listdir(sample)
        views = [os.path.join(sample, view) for view in views]
        videos = []
        frame_names = []
        for view in views:
            video = self.crop_rotate_view(view, True)
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
        print('Saved:  ', sample)

    def crop_rotate_view(self, view, rotate):
        # Takes view folder and returns all the cropped images as array
        # Takes argument rotate as boolean to check if you want to rotate it
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
        h = int(self.crop_height/8)
        w = int(self.crop_width/8)
        if view1:
            img = cv.resize(img, (w,h))
        else:
            img = cv.resize(img, (h,w))
            img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
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
        return img[starty:starty+cropy,startx:startx+cropx]

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

    def rotate(self, sample):
        frames = os.listdir(sample)
        frames = [os.path.join(sample, frame) for frame in frames]
        for frame in frames:
            img = cv.imread(frame)
            img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
            frame = frame.replace(self.dataset_path, self.save_path)
            folder = os.path.dirname(frame)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            cv.imwrite(frame, img)
        print('Saved:  ', sample)

    def pipeline(self, sample):
        self.crop_resize_concatenate(sample)

    def start(self):
        if __name__ == '__main__':
            # Dataset processing
            views = self.get_sample_folder_number(self.dataset_path)
            self.pipeline(views[0])
            try:
                #use crop_resize_concatenate for resizing concatenating and saving
                #use rotate to rotate images
                pool = Pool(5)      # Create a multiprocessing Pool.
                pool.map(self.rotate, views)  # process data_inputs iterable with pool
            finally: # To make sure processes are closed in the end, even if errors happen
                pool.close()
                pool.join()
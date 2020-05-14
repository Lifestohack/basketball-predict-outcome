from multiprocessing import Pool
import os
import cv2 as cv
import numpy as np
import denseflow

class DataMultiProcess():
    def __init__(self, dataset_path, save_path, of_save_path, create_dense, resize):
        super().__init__()
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.of_save_path = of_save_path
        self.create_dense = create_dense
        self.resize = resize
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
        h = int(self.crop_height/6)
        w = int(self.crop_width/6)
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

    def pipelines(self, sample):
        if self.create_dense == True:
            print("Creating dense Images.")
        views = os.listdir(sample)
        views = [os.path.join(sample, view) for view in views]
        
        videos = []

        for view in views:
            frames = os.listdir(view)
            frames = [os.path.join(view, frame) for frame in frames]
            video = []
            # Read images from view starts here #
            for frame in frames:
                img = cv.imread(frame)
                video.append(img)
            # Read images from view ends here #

            # remove the background of the previously read view starts here #
            video = self.remove_background(video)
            # remove the background of the previously read view starts here #
        
            # Dense Video starts here #
            if self.create_dense == True:
                video = self.dense_video(video)
            # Dense Video end here #

            # Crop resize and rotate if view 2 starts here # 
            view1 = False
            width = self.crop_width                 # view1 is horizontal video
            height = self.crop_height
            if "view1" in view:
                view1 = True
            else:
                view1 = False
                width = self.crop_height            # view2 is vertical video so width and height changes
                height = self.crop_width
            
            crop_resize_rotate_if_view2_video = []
            for frame in video:
                img = self._crop_image(frame, width, height, view1)
                img = self.resize_rotate(img, view1)        # rotate only if view 2
                crop_resize_rotate_if_view2_video.append(img)
            # Crop resize and rotate if view 2 ends here # 
            
            videos.append(crop_resize_rotate_if_view2_video)

        # Concatenation of two views starts here #
        combo_video = np.concatenate(videos, axis=1)
        # Concatenation of two views starts here #

        # Rotate starts here #
        ROTATE_90_COUNTERCLOCKWISE = self.rotate_video(combo_video, cv.ROTATE_90_COUNTERCLOCKWISE)
        ROTATE_90_CLOCKWISE = self.rotate_video(combo_video, cv.ROTATE_90_CLOCKWISE)
        ROTATE_180 = self.rotate_video(combo_video, cv.ROTATE_180)
        # Rotate ends here #
        

        # Saving videos starts here #
        frames = [path.replace(self.dataset_path, self.save_path) for path in frames] # remove data_set path to save_path
        frames = [path.replace("view1\\", "") for path in frames] # remove view1
        frames = [path.replace("view2\\", "") for path in frames] # remove view2
        self.save_video(combo_video, frames)


        save_rotate_path_90_counterclockwise = self.save_path + '_rotate_90_counterclockwise'
        save_rotate_path_90_counterclockwise = [path.replace(self.save_path , save_rotate_path_90_counterclockwise) for path in frames] 
        self.save_video(ROTATE_90_COUNTERCLOCKWISE, save_rotate_path_90_counterclockwise)
        
        save_rotate_path_90_clockwise = self.save_path + '_rotate_90_clockwise'
        save_rotate_path_90_clockwise = [path.replace(self.save_path , save_rotate_path_90_clockwise) for path in frames] 
        self.save_video(ROTATE_90_CLOCKWISE, save_rotate_path_90_clockwise)

        save_rotate_path_180 = self.save_path + '_rotate_180'
        save_rotate_path_180 = [path.replace(self.save_path , save_rotate_path_180) for path in frames] 
        self.save_video(ROTATE_180, save_rotate_path_180)
        # Saving videos Ends here #

    def remove_background(self, video, rate=30):                                       
        background = np.median(video, axis=0, keepdims=True).astype(np.uint8)
        foreground = np.absolute((background - video))    
        foreground[foreground > (255 - rate)] = 0
        foreground[foreground < (0 + rate)] = 0          
        #Image.fromarray(foreground[0]).save('result.png')
        return foreground
    
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
            # Dataset processing
            views = self.get_sample_folder_number(self.dataset_path)
            #self.pipelines(views[0])
            try:
                # use crop_resize_concatenate for resizing concatenating and saving
                # use rotate to rotate images
                pool = Pool()                  # Create a multiprocessing Pool.
                pool.map(self.pipelines, views)    # process data_inputs iterable with pool
                pass
            finally:                            # To make sure processes are closed in the end, even if errors happen
                #pool.close()
                #pool.join()
                pass

DataMultiProcess('orgdata', 'data\\no_background\\crop_resize_concatenate_128x128', 'optics', True, (128, 128)).start()

import os
import cv2 as cv

def recursive_prepare_data(data_path):
    filespath = []
    subdir = os.listdir(data_path)
    for subpath in subdir:
        subsubpath = os.path.join(data_path, subpath)
        if os.path.isfile(subsubpath):
            filespath.append(subsubpath)
        else:
            subsub_path_list = recursive_prepare_data(subsubpath)
            if len(subsub_path_list) > 0:
                crop(subsub_path_list)
    return filespath

def crop(frames):
    width = 780
    height = 550
    path = frames[0].replace(readpath, savepath)
    savedir = os.path.dirname(path)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    else:
        print('Folder available in ', savedir)
        return
    view1=False
    if 'view1' in path:
        view1 = True
    for frame in frames:
        img = cv.imread(frame)
        if view1:
            img = crop_center(img, width, height)    # W x H
        else:
            img = crop_center(img, height, width)    # W x H
        frame = frame.replace(readpath, savepath)
        #frame = frame.replace('.png', '.jpg')
        cv.imwrite(frame, img)
    print('Done for ', savedir)

def crop_center(img, cropx, cropy):
    y,x,z = img.shape
    startx = x//2-(cropx//2) - 40
    starty = y//2-(cropy//2) + 40 
    return img[starty:starty+cropy,startx:startx+cropx]

readpath = 'data'
savepath = 'cropjpg'
recursive_prepare_data(readpath)
print('Done!!!')
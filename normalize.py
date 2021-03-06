#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, csv
import configparser
from PIL import Image
import torchvision, torch
import pandas as pd

def get_sample_folder_number(data_path):
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
                subsub_path_list = get_sample_folder_number(subsubpath)   
            filespath += subsub_path_list                
    return filespath

def getsample(sample, trajectory):

    frames = os.listdir(sample)
    if "view" in frames[0]:
        viewcsv = []
        for view in frames:
            path_to_read = os.path.join(sample, view)
            frames = os.listdir(path_to_read)
            csvpath = os.path.join(path_to_read, frames[0])
            temp = pd.read_csv(csvpath, header=None)
            temp_tensor = torch.tensor(temp.values)
            viewcsv.append(temp_tensor)
        stack_view = torch.cat(viewcsv)
        return stack_view
    else:
        return getframes(frames, sample, trajectory)



def getframes(frames, sample, trajectory):
    frames_data = []
    for idx, frame in enumerate(frames):
        path_to_read = os.path.join(sample, frame)
        img = None
        if trajectory == True:
            pass
        else:
            img = Image.open(path_to_read)
            img = torchvision.transforms.ToTensor()(img)
        frames_data.append(img)
    return torch.stack(frames_data)


def calmeanstd(path, trajectory):
    # Config parser
    config = configparser.ConfigParser()
    config.read('config.ini')
    config = config['DEFAULT']   
    meanstd = config['meanstd']
    optic = config['optic']
    save_path = path + ".csv"
    #save_path = os.path.join(os.path.split(path)[0], meanstd)
    if os.path.exists(save_path):
        return readMeanSTDcsv(save_path, trajectory)
    samples = get_sample_folder_number(path)
    mean = 0.
    std = 0.
    nb_samples = 0
    csvmeanstd = []
    print("No Mean and Standard deviation value found.")
    for sample in samples:
        data = getsample(sample, trajectory)
        if trajectory == True:
            data = data.view(-1)
            mean = data.mean().numpy()
            std = data.std().numpy()
        else:
            data = data.view(1, data.size(1), -1)
            mean = data.mean(2).sum(0).numpy()
            std = data.std(2).sum(0).numpy()
        nb_samples += 1
        outstr = 'Calculating mean and standard deviation: {}/{}'.format(nb_samples, len(samples))
        sys.stdout.write('\r'+ outstr)
        if trajectory == True:
            csvmeanstd.append((sample, mean, std))
        else:
            csvmeanstd.append((sample, mean[0], mean[1], mean[2], std[0], std[1], std[2]))
    exportMeanSTDcsv(csvmeanstd, save_path)
    print("")
    return readMeanSTDcsv(save_path, trajectory)
    

def exportMeanSTDcsv(data, save_path):
    with open(save_path, 'w', newline='') as f:
        fw = csv.writer(f, delimiter=',')
        fw.writerows(data)

def readMeanSTDcsv(path, trajectory):
    data = {}
    with open(path, newline='') as csvfile:
        for row in csv.reader(csvfile):
            if trajectory == True:
                 data[row[0]] = float(row[1]), float(row[2])           
            else:
                data[row[0]] = float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])
    return data
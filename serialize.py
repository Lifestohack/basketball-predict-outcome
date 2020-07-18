#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from datetime import datetime
import os
import csv
import configparser
import helper as helper

def save_module(model, modelclass, path):
    if model is None or modelclass is None or path is None:
        raise RuntimeError("model, class name or path can not be empty")
    if not os.path.exists(path):
        os.makedirs(path)
    timestamp = __get_timestamp()
    save_name = modelclass + '_' + timestamp + '.pt'
    save_path = os.path.join(path, save_name)
    torch.save(model.state_dict(), save_path)
    return save_path

def load_module(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def save_results(results, modelclass, path):
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = modelclass + '_' + __get_timestamp() + '.csv'
    save_path = os.path.join(path, save_name)
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)
    return save_path

def load_results(path):
    results = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for result in reader:
            results.append(result)
    return results

def get_all_results_names():
    # Config parser
    config = configparser.ConfigParser()
    config.read('config.ini')
    config = config['DEFAULT']
    save_path = config['output']
    result_folder = config['results']
    save_path_results = os.path.join(save_path)
    result_list = helper.get_all_files(save_path_results)
    result_list = [x for x in result_list if 'results' in x]
    result_list = [x for x in result_list if '.directory' not in x]
    return result_list

def __get_timestamp():
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    return timestamp

def exportcsv(dicthitormiss, modelclass, path):
    if not os.path.exists(path):
        os.makedirs(path)
    fname = modelclass + '_' + __get_timestamp() + '.csv'
    save_path = os.path.join(path, fname)
    fieldnames = ['id', 'category']
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        for key, value in dicthitormiss.items():
            writer.writerow({'id': key, 'category': value})
    return save_path
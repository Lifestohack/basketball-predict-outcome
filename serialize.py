import torch
from datetime import datetime
import os
import csv
import configparser

def save_module(model, modelclass, path):
    if model is None or modelclass is None or path is None:
        raise RuntimeError("model, class name or path can not be empty")
    if not os.path.exists(path):
        os.mkdir(path)
    timestamp = __get_timestamp()
    save_name = modelclass + '_' + timestamp + '.pt'
    save_path = os.path.join(path, save_name)
    torch.save(model, save_path)
    return timestamp

def load_module(CLASS, path):
    model = CLASS()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def save_results(results, path):
    if not os.path.exists(path):
        os.mkdir(path)
    save_name = __get_timestamp() + '.csv'
    save_path = os.path.join(path, save_name)
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)

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
    save_path_trained_network = config['trained_network']
    result_folder = config['results']
    save_path_results = os.path.join(save_path_trained_network, result_folder)
    result_list = os.listdir(save_path_results)
    result_list = [os.path.join(save_path_results, result)  for result in result_list]
    return result_list

def __get_timestamp():
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    return timestamp
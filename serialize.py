import torch
from datetime import datetime
import os
import csv

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

def __get_timestamp():
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    return timestamp

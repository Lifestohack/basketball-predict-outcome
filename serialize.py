import torch
from datetime import datetime
import os

class Serialize():
    def __init__(self, path):
        super().__init__()
        self.path = path
        if not os.path.isdir(self.path):
                os.makedirs(self.path)

    def save(self, model, modelclass):
        if model is None or modelclass is None:
            raise RuntimeError("model or class name can not be empty")
        now = datetime.now()
        timestamp = modelclass + '_' + now.strftime("%Y_%m_%d_%H_%M_%S") + '.pt'
        path = os.path.join(self.path, timestamp)
        torch.save(model, path)

    def load(self, CLASS, path):
        model = CLASS()
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

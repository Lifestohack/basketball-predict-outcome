import torch
from datetime import datetime
import os

class serialize():
    def __init__(self, path):
        super().__init__()
        self.path = path

    def save(self, model, modelclass):
        if model is None or modelclass is None:
            raise RuntimeError("model or class name can not be empty")
        now = datetime.now()
        timestamp = modelclass + now.strftime("%Y_%m_%d_%H_%M_%S") + '.pt'
        path = os.path.join(self.path, timestamp)
        torch.save(model, self.path)

    def load(self, CLASS, path):
        model = CLASS()
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

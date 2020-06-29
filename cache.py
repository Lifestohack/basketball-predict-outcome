from psutil import virtual_memory

class Cache():
    def __init__(self):
        super().__init__()
        self.__THRESHOLD = 1536 * 1024 * 1024 # in byte
        self.__length = 0
        self.__cache = {}

    def getcache(self, sample):
        cache = self.__cache.get(sample)
        if cache == None:
            return None, False
        else:
            return cache, True

    def setcache(self, path, frames, label):
        mem = virtual_memory().available
        if mem <= self.__THRESHOLD:
            return
        cache = self.__cache.get(path)
        if cache == None:
            self.__length += 1
            value = [frames, label]
            self.__cache[path] = value #'dataset/data/crop_resize_concatenate_128x128/training/hit/9'
            pass

    def getlength(self):
        return __length
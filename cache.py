#!/usr/bin/env python

from psutil import virtual_memory

class Cache():
    # Window memory management paging doesnot seems to handle this properly. 
    # After running the training for long time, it starts paging the memory
    # When it is again needed then it takes long time to retrieve
    # Run it in a linux operating system if that problem persist
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
            self.__cache[path] = value 

    def getlength(self):
        return __length
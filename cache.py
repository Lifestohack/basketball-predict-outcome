#!/usr/bin/python

from psutil import virtual_memory

class Cache(object):
    class __Cache:
        def __init__(self):
            self.__THRESHOLD = 0.5  # in GB
            self.__length = 0
            self.__c = {}

        def getcache(self, sample):
            c = self.__c.get(sample)
            if c == None:
                return (None, False)
            else:
                return (c, True)

        def setcache(self, path, frames, label):
            c = self.__c.get(path)
            if c == None:
                mem = (virtual_memory().available)/(1024 * 1024 * 1024)
                if mem <= self.__THRESHOLD:
                    return
                self.__length += 1
                value = [frames, label]
                self.__c[path] = value

        def getlength(self):
            return __length

    instance = None
    def __new__(cls): # __new__ always a classmethod
        if not Cache.instance:
            Cache.instance = Cache.__Cache()
        return Cache.instance
    def __getattr__(self, name):
        return getattr(self.instance, name)
    def __setattr__(self, name):
        return setattr(self.instance, name)
    
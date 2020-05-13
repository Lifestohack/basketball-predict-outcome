from psutil import virtual_memory

class Cache():
    def __init__(self):
        super().__init__()
        self.__THRESHOLD = 1024 * 1024 * 1024 # in byte
        self.__length = 0
        self.__cache = []

    def getcache(self, sample):
        cache = [item for item in self.__cache if sample == item['path']]
        if len(cache) == 0:
            return None, False
        else:
            return cache[0], True


    def setcache(self, path, frames, label):
        __mem = virtual_memory()
        if __mem.free <= self.__THRESHOLD:
            return
        self.__length += 1
        dictsframes = {'path': path,
                'frames': frames,
                'label':label}
        self.__cache.append(dictsframes)

    def getlength(self):
        return __length
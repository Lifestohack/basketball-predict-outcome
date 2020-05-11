from multiprocessing import Pool
import Dataprocess
import os

class DataMultiProcess():
    def __init__(self, dataset_path, save_path, of_save_path, pp_opt_flow, resize):
        super().__init__()
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.of_save_path = of_save_path
        self.pp_opt_flow = pp_opt_flow
        self.resize = resize

        self.process = Dataprocess.Preprocess(dataset_path=self.dataset_path, 
                            save_path=self.save_path, 
                            of_save_path=self.of_save_path, 
                            pp_opt_flow=self.pp_opt_flow, 
                            resize=self.resize)

    def get_views(self, data_path):
        go_deeper = True
        filespath = []
        subdir = os.listdir(data_path)
        for subpath in subdir:
            subsubpath = os.path.join(data_path, subpath)
            if 'view' in subsubpath:
                go_deeper = False
                filespath.append(subsubpath)
            else:
                if go_deeper:
                    subsub_path_list = self.get_views(subsubpath)   
                filespath += subsub_path_list                
        return filespath

    def process_view(self, view):
        frames_path = os.listdir(view)
        frames_path = [os.path.join(view, frame) for frame in frames_path ]
        self.process.data_process(frames_path)

    def start(self):
        if __name__ == '__main__':
            # Dataset processing
            views = self.get_views(dataset_path)
            try:
                pool = Pool()                         # Create a multiprocessing Pool
                pool.map(self.process_view, views)  # process data_inputs iterable with pool
            finally: # To make sure processes are closed in the end, even if errors happen
                pool.close()
                pool.join()

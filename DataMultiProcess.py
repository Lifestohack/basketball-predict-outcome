from multiprocessing import Pool
import Dataprocess
import os

dataset_path = 'data'
save_path = 'cache'
of_save_path = 'optics'
pp_opt_flow = True
resize = (50,50)

process = Dataprocess.Preprocess(dataset_path=dataset_path, 
                    save_path=save_path, 
                    of_save_path=of_save_path, 
                    pp_opt_flow=pp_opt_flow, 
                    resize=resize)

def get_views(data_path):
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
                subsub_path_list = get_views(subsubpath)   
            filespath += subsub_path_list                
    return filespath

def process_view(view):
    frames_path = os.listdir(view)
    frames_path = [os.path.join(view, frame) for frame in frames_path ]
    process.data_process(frames_path)

if __name__ == '__main__':
    # Dataset processing
    views = get_views(dataset_path)
    try:
        pool = Pool()                         # Create a multiprocessing Pool
        pool.map(process_view, views)  # process data_inputs iterable with pool
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()
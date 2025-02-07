import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from read_video import *
import threading

videoreader = VideoReader(verbose=False)
video_read_fn = lambda x: videoreader.read_frames(x, num_frames=10)

def task_ext_frame(task_n, sp_meta):
    data_root = 'data/split/'+task_n
    out_root = "data/frames/"+task_n
    
    for i, r in sp_meta.iterrows(): 
        split = r['split']
        name = r['name']
        
        try:
            fpath = data_root + '/' + split + '/' + name
            my_frames, my_idxs = video_read_fn(fpath)

            out_dir = out_root + '/' + split + '/' + name.replace('.mp4','') # TODO: watch out file name not end with .mp4
            print(f'out_dir={out_dir}')
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            for i, fm in enumerate(my_frames):
                fm = cv2.cvtColor(fm, cv2.COLOR_BGR2RGB)
                fimg = Image.fromarray(fm).convert('RGB')
                fimg.save(f"{out_dir}/{i}.png")
        except Exception as e:
            print(f'Exception: task_n={task_n} split={split}, name={name}, e={e}')
            
def thr_task(task_n, thr_cnt):
    my_meta = pd.read_csv('data/split/'+task_n+'/metadata.csv')
    
    check_meta(my_meta)
    
    df_split = np.array_split(my_meta, thr_cnt)
    
    threads = []
    for i in range(thr_cnt):
        threads.append(threading.Thread(target = task_ext_frame, args = (task_n, df_split[i],) ))
        threads[i].start()

    for i in range(thr_cnt):
        threads[i].join()
        
    print('all thread done')

def check_meta(xmeta):
    # 1: video end name
    for i,r in xmeta.iterrows():
        name = r['name']
        assert '.mp4' in name
        
    # 2: split name
    spl = list(xmeta['split'].unique())
    assert 'train' in spl
    assert 'test' in spl
    assert 'valid' in spl
    
#===========MAIN===========
task_name_l = ['yt_cut_new']

task_exe_time = []
for tk_name in task_name_l:
    
    start_time = time.time()
    #------------
    thr_task(tk_name, 4)  # TODO: change HERE
    #------------
    task_time = int(time.time() - start_time)
    print(f'tk_name={tk_name}, task_time={task_time} sec')
    print()
    
    task_exe_time.append([tk_name, task_time])
    
task_exe_time = pd.DataFrame(task_exe_time, columns=['task_name', 'task_time'])
task_exe_time.to_csv('output/ext_frame_time_yt_cut_new.csv')
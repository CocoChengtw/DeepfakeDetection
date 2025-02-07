import os
import subprocess
import pandas as pd
import json
import shutil
import numpy as np
import time
import tarfile


def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def dump_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)

def remove_dot_png(x):
    new_x = x.replace(".png","").replace(".","") + ".png"
    return new_x

def load_meta(file_path, stage):
    df = pd.read_csv(file_path)
    df.index = df["name"].apply(lambda x: x.replace("'", ""))
    df = df[df["split"]==stage]
    df = df.drop(["name", "split"], axis=1)
    return df

def dict_replace_nan(dic):
    new_dic = {}
    for k, v in dic.items():
        if v is np.nan:
            new_dic[k] = str(None)
        else:
            new_dic[k] = v
    
    return new_dic

def run_cmd(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        print(e.stderr.decode())

#------MAIN-------
DATASET = ['yt_cut_new']
STAGE = ['train', 'test', 'valid']

for data in DATASET:
    for st in STAGE:
        temp_data = os.path.join(f"data/webdataset_tmp/{data}")
        temp_data_folder = os.path.join(f"data/webdataset_tmp/{data}", st)
        input_image_folder = f"data/faces_retina/{data}/{st}"

        if not os.path.exists(temp_data):
            os.mkdir(temp_data)
        if not os.path.exists(temp_data_folder):
            os.mkdir(temp_data_folder)

        video_names = os.listdir(input_image_folder)
        chunk_size = 1000
        video_sublist = split_list(video_names, chunk_size=chunk_size)
        print(data, 'ttl tar:', len(video_sublist))
        meta = load_meta(f"data/split/{data}/metadata.csv", stage=st)

        # iterate video
        start_time = time.time()
        for i, sublist in enumerate(video_sublist):
            # Create tar folder
            tar_folder_path = os.path.join(temp_data_folder, f"{st}_{i}")
            if not os.path.exists(tar_folder_path):
                os.mkdir(tar_folder_path)
            print(f"TAR {i} th")

            # Iterate videos
            for j, video_name in enumerate(sublist):
                if 'ipynb_checkpoints' not in video_name:
                    video_meta = meta.loc[f"{video_name}.mp4"].to_dict()
                    video_meta = dict_replace_nan(video_meta)
                    video_folder_path = os.path.join(input_image_folder, video_name)

                # Iterate images
                for image_filename in os.listdir(video_folder_path):

                    if image_filename.endswith(".png"):
                        input_image_file_path = os.path.join(video_folder_path, image_filename)
                        fix_image_filename = remove_dot_png(video_name + '_' + image_filename)
                        output_image_file_path = os.path.join(tar_folder_path, fix_image_filename)
                        # Copy image
                        shutil.copyfile(input_image_file_path, output_image_file_path)
                        # Create json for image
                        new_meta_path = os.path.join(tar_folder_path, fix_image_filename.replace(".png", ".json"))
                        dump_json(new_meta_path, video_meta)

                if (j+1)%100==0:
                    print(f"{j} th file finish processing......")

        task_time = int(time.time() - start_time)
        print(data, st, '; iterate video task time:', task_time, 'sec')

        
        # webdataset
        print('webdataset start', data, st)
        data_folder = os.path.join(f"data/webdataset/{data}")
        output_data_folder = os.path.join(f"data/webdataset/{data}", st)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        if not os.path.exists(output_data_folder):
            os.mkdir(output_data_folder)

        start_time = time.time()
        
        for tar_folder_name in os.listdir(temp_data_folder):
            source_tar_folder_path = os.path.join(temp_data_folder, tar_folder_name)
            des_tar_file_path = os.path.join(output_data_folder, f"{tar_folder_name}.tar")

            with tarfile.open(des_tar_file_path, "w:gz") as tar:
                for file in sorted(os.listdir(source_tar_folder_path)):  # 顯式排序
                    # print(f"Processing folder: {file}")
                    filepath = os.path.join(source_tar_folder_path, file)
                    tar.add(filepath, arcname=os.path.basename(file))

        task_time = int(time.time() - start_time)
        print(data, st, '; webdataset task time:', task_time, 'sec')
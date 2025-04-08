"""
Uses ONNX-based face detector to extract faces from video frames.
Processes each frame in [train, test, valid] and saves cropped faces.
"""

import onnxruntime
from deepfake_utils import *

from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel
import pandas as pd
import math
import random
import os
import numpy as np
import glob

from PIL import Image
import cv2

import subprocess, sys
import time


# run command to load sample dataset
def run_cmd(cmd):
    try:
        result = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT
        )  # shell = True, executable = "/bin/bash",
    except subprocess.CalledProcessError as cpe:
        result = cpe.output
        print("Exception, result=", result)
        raise

    return result


face_ext_path = "model/FaceDetector.onnx"
faceExt = MyReinfaceOnnx(face_ext_path)


def ext_face_and_detect(im_arr):
    fimg = faceExt.extract_face(im_arr)
    return fimg


def task_ext_face(dataset, split):

    dataset_videos = os.listdir(FRAME_ROOT + "/" + dataset + "/" + split)
    toal = len(dataset_videos)
    cnt = 0
    for i, v in enumerate(dataset_videos):

        cnt += 1
        if cnt % 10 == 0:
            print(f" progress={cnt}/{toal}")

        try:
            src_fd_path = FRAME_ROOT + "/" + dataset + "/" + split + "/" + v
            out_fd_path = FACE_ROOT + "/" + dataset + "/" + split + "/" + v

            if os.path.exists(out_fd_path):
                res = run_cmd(f"rm -r '{out_fd_path}'")
            res = run_cmd(f"mkdir '{out_fd_path}'")
            # print(f'mkdir={out_fd_path}, res={res}')
        except Exception as e:
            print(f"<Eception>: {out_fd_path} Eception e={e}")
            continue

        img_l = os.listdir(src_fd_path)
        for im_n in img_l:
            try:
                im = Image.open(src_fd_path + "/" + im_n)
                im_arr = np.asarray(im)
                fimg = faceExt.extract_face(im_arr)
                if fimg is None:
                    print(f"<NO FACE>: {out_fd_path}/{im_n} ")
                else:
                    fimg = Image.fromarray(fimg).convert("RGB")
                    fimg.save(f"{out_fd_path}/{im_n}")
            except Exception as e:
                print(f"<Eception>: {out_fd_path}/{im_n} Eception e={e}")


# ------MAIN-------
FRAME_ROOT = "data/frames"
FACE_ROOT = "data/faces_retina"

task_exe_time = []
dataset_l = [""]

split_lst = ["train", "test", "valid"]
for dataset in dataset_l:
    if not os.path.exists(FACE_ROOT + "/" + dataset):
        os.mkdir(FACE_ROOT + "/" + dataset)
    for split in split_lst:
        if not os.path.exists(FACE_ROOT + "/" + dataset + "/" + split):
            os.mkdir(FACE_ROOT + "/" + dataset + "/" + split)
        print(f" dataset={dataset}, split={split} start")

        start_time = time.time()
        # ------------
        task_ext_face(dataset, split)
        # ------------
        task_time = int(time.time() - start_time)
        print(f"dataset={dataset}, split={split}, task_time={task_time} sec")
        print()

    task_exe_time.append([dataset, task_time])

task_exe_time = pd.DataFrame(task_exe_time, columns=["dataset", "task_time"])
task_exe_time.to_csv("output/ext_face_time.csv")

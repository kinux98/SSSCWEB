from __future__ import print_function, absolute_import
from configparser import Interpolation

import os
import random
from struct import unpack
from tracemalloc import start
from typing import Union
import numpy as np
import math
import cv2
import torch
import time
from matplotlib import cm
import json
from os.path import join as pjn
from PIL import Image
from torchvision import transforms
from torch.nn.functional import interpolate
import albumentations as A
from tqdm import tqdm
from data.keypoint_to_flow import KeypointToFlow

import json


def _yt_init(image_set, json_file, pl_path):
    save_dict = {}
    
    prefix_list           = []
    src_path_pred_list    = []
    tgt_path_pred_list    = []
    src_path_img_list     = []
    tgt_path_img_list     = []
    yt_video = 0
    yt_shot = 0
    yt_frame = 0
    yt_pair = 0
    print("loading data path..")
    
    candid_class_name = os.listdir(pl_path)
    with open(json_file) as jfile:
        data = json.load(jfile)
    for s_class in tqdm(candid_class_name, total=len(candid_class_name)):
        ytid_path = pjn(pl_path, s_class)
        for s_ytid in os.listdir(ytid_path):
                shot_path = pjn(pl_path,s_class,s_ytid)
                yt_video += 1
                for s_shot in os.listdir(shot_path):
                    yt_shot += 1
                
                    pred_list = pjn(pl_path,s_class,s_ytid, s_shot, "pred")
                    # print(pred_list)
                    coord_list = pjn(pl_path,s_class,s_ytid, s_shot, "coord")

                    # assert os.path.exists(os.path.join(pred_list, "0_pred.npy"))== True
                    # assert os.path.exists(os.path.join(coord_list, "0_coord.npy"))== True
                    tl = len(os.listdir(pred_list))
                    candid_idx = [i for i in range(tl)] # [0, 1, 2, ..., 16] : 17
                    del candid_idx[0]

                    # [1,2,3,4,5,...,16]
                    start_idx = data[s_class][s_ytid][s_shot][0]

                    src_path_pred = os.path.join(pred_list, "0_pred.npy")
                    src_path_coord = os.path.join(coord_list, "0_coord.npy")

                    single_shot = pjn(image_set, s_class, s_ytid, "frames")
                    candid_sufix = "real_"+"{0:010d}".format(start_idx)+"_save_"+"{0:010d}".format(start_idx)+".png"
                    # print(candid_sufix, start_idx)
                    src_path_img = os.path.join(single_shot, candid_sufix)
                    yt_frame += 1

                    # assert os.path.exists(src_path_img) == True
                    # print(candid_idx)
                    for si in candid_idx:
                        tgt_path_pred = pjn(pred_list, "%s_pred.npy"%(str(si)))

                        pic_name = "real_"+"{0:010d}".format(si+start_idx)+"_save_"+"{0:010d}".format(si+start_idx)+".png"
                        tgt_path_img = os.path.join(single_shot, pic_name)
                        yt_frame += 1
                        yt_pair += 1

                        prefix_list.append(single_shot)

                        src_path_pred_list.append(src_path_pred)
                        tgt_path_pred_list.append(tgt_path_pred  )
                        
                        src_path_img_list.append(src_path_img )
                        tgt_path_img_list.append(tgt_path_img)
    
    save_dict['src_path_pred_list'] = src_path_pred_list
    save_dict['tgt_path_pred_lis'] = tgt_path_pred_list
    save_dict['src_path_img_list'] = src_path_img_list
    save_dict['tgt_path_img_list'] =  tgt_path_img_list
    
    with open('./file_list.json','w') as f:
        json.dump(save_dict, f, ensure_ascii=False, indent=4)
    

_yt_init(image_set="/home/kinux98/videos_thumbnail_new",
        json_file="./video_scene_parsing_new_mt.json",
        pl_path="/data01/kinux98/results")


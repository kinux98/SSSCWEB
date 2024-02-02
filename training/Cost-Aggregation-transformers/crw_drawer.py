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
import random

class YoutubeDataset(torch.utils.data.Dataset):
    def __init__(self, image_set, json_file, pl_path):

        self._yt_init(image_set, json_file, pl_path)
        self.imside = 256
        
        self.max_pts = 256*256

        random_transform = transforms.RandomApply([
            transforms.ColorJitter(0.1,0.1,0.1,0.1)
        ], p=0.2)

        self.img_transform_src = transforms.Compose([
            transforms.Resize((self.imside, self.imside)),
            transforms.RandomGrayscale(0.1),
            random_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        self.img_transform_tgt = transforms.Compose([
            transforms.Resize((self.imside, self.imside)),
            transforms.RandomGrayscale(0.1),
            random_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        self.kps_to_flow = KeypointToFlow(receptive_field_size=35, jsz=256//16, feat_size=16, img_size=self.imside)
 
    def get_image(self, img_path):
        r"""Reads PIL image from path"""
        path = os.path.join(img_path)
        return Image.open(path).convert('RGB') 

    def get_points(self, pts_list):
        r"""Returns key-points of an image with size of (240,240)"""
        xy, n_pts = pts_list.size() # 2, number of points

        pad_pts = torch.zeros((xy, self.max_pts - n_pts)) - 1
        x_crds = pts_list[0]
        y_crds = pts_list[1]
        kps = torch.cat([torch.stack([x_crds, y_crds]), pad_pts], dim=1)
        
        return kps, n_pts

    def __getitem__(self, index):
        src_path_pred = self.src_path_pred_list[index]
        tgt_path_pred = self.tgt_path_pred_list[index]

        src_path_img = self.src_path_img_list[index]
        tgt_path_img = self.tgt_path_img_list[index]
        
        src_pred = torch.from_numpy(np.load(src_path_pred))
        # print(src_pred.shape)
        tgt_pred = torch.from_numpy(np.load(tgt_path_pred))

        src_img_pil = self.get_image(src_path_img)
        src_img = self.img_transform_src(src_img_pil)

        tgt_img_pil = self.get_image(tgt_path_img)
        tgt_img = self.img_transform_tgt(tgt_img_pil)

        target_idx = np.intersect1d(np.unique(tgt_pred), np.unique(src_pred))
        target_idx = target_idx[target_idx!=-99]
        
        return (
            src_img, src_pred,
            tgt_img, tgt_pred,
        )

    def __len__(self):
        return self.len

    def _yt_init(self, image_set, json_file, pl_path):
        
        prefix_list           = []
        src_path_pred_list    = []
        # src_path_coord_list   = []
        tgt_path_pred_list    = []
        # tgt_path_coord_list   = []
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
                            tgt_path_coord = pjn(coord_list, "%s_coord.npy"%(str(si)))
    
                            pic_name = "real_"+"{0:010d}".format(si+start_idx)+"_save_"+"{0:010d}".format(si+start_idx)+".png"
                            # print(s_class, s_ytid, s_shot, pic_name, si, start_idx)
                            tgt_path_img = os.path.join(single_shot, pic_name)

                            yt_frame += 1

                            yt_pair += 1
    
                            prefix_list.append(single_shot)
    
                            src_path_pred_list.append(src_path_pred)
                            tgt_path_pred_list.append(tgt_path_pred  )
                            
                            src_path_img_list.append(src_path_img )
                            tgt_path_img_list.append(tgt_path_img)
        self.src_path_pred_list = src_path_pred_list
        self.tgt_path_pred_list = tgt_path_pred_list
        self.src_path_img_list = src_path_img_list
        self.tgt_path_img_list = tgt_path_img_list
        self.len = len(self.src_path_img_list)
        print("Done.")
        print("# of yt videos : ", yt_video)
        print("# of yt shots : ", yt_shot)
        print("# of yt pairs : ", yt_pair)
        print("# of yt frames : ", yt_frame)
        # exit(1)

def imwrite(path, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)

def drawing(dataset, idx_list):
    outpath = os.path.join("./crw_save")
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    for idx in idx_list:
        src_img, src_pred, tgt_img, tgt_pred, = dataset[idx]

        target_idx = np.intersect1d(np.unique(tgt_pred), np.unique(src_pred))
        target_idx = target_idx[target_idx!=-99]

        print(len(target_idx))
        if len(target_idx) > 15:
            continue        
        
        source_image_ = (torch.from_numpy(src_img)).squeeze() # h x w x 3
        target_image_ = (torch.from_numpy(tgt_img)).squeeze() # h x w x 3
        concat_image = torch.hstack([source_image_,  target_image_]).numpy() # h x 2w x 3

        for ccnt, rnd_sampled_idx in enumerate(target_idx):
            k = torch.nonzero(torch.from_numpy(src_pred) == rnd_sampled_idx,as_tuple=False)
            base_y, base_x = k[0]
            base_y = base_y.item()
            base_x = base_x.item()
            base_y = int(base_y)
            base_x = int(base_x)
            corr_candid = torch.nonzero(tgt_pred == rnd_sampled_idx, as_tuple=False)
            
            for cccnt, rnd_sampled_idx_corr in enumerate(corr_candid):
                corr_coord = corr_candid[rnd_sampled_idx_corr]
                wrpd_y = corr_coord[0].item()
                wrpd_x = corr_coord[1].item()
                wrpd_y = int(wrpd_y)
                wrpd_x = int(wrpd_x)
                dest_coord = (base_x, base_y)
                source_coord = (wrpd_x+256, wrpd_y)
                # print(source_coord, dest_coord)
                cv2.line(concat_image, source_coord, dest_coord, 'g', thickness=1, lineType=cv2.LINE_AA)

        final_path = os.path.join(outpath, "concat_%d.jpg"%(idx))
        print(final_path)
        imwrite(final_path, concat_image)


if __name__ == "__main__":
    youtube_dataset = YoutubeDataset(
        image_set="/home/kinux98/videos_thumbnail_new",
        json_file="./video_scene_parsing_new_mt.json",
        pl_path="/data01/kinux98/results",
    )

    r_idx = random.sample([i for i in range(10000)], 10)
    


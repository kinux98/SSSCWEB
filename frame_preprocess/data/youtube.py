from __future__ import print_function, absolute_import

import os
from tracemalloc import start
from typing import Union
import numpy as np
import math
import cv2
import torch
import time
from matplotlib import cm
import json


def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)


def sequence_to_string(seq: np.ndarray) -> str:
    return "".join([chr(c) for c in seq])


def pack_sequences(seqs: Union[np.ndarray, list]):
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in seqs])
    return values, offsets


def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    return img

def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    img = cv2.resize( img, (owidth, oheight) )
    img = im_to_torch(img)
    return img

def load_image(img_path):
    # H x W x C => C x H x W
    img = cv2.imread(img_path)
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:,:,::-1]
    img = img.copy()
    return im_to_torch(img)

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

######################################################################
def try_np_load(p):
    try:
        return np.load(p)
    except:
        return None

def make_lbl_set(lbls):
    lbl_set = [np.zeros(3).astype(np.uint8)]
    count_lbls = [0]    
    
    flat_lbls_0 = lbls[0].copy().reshape(-1, lbls.shape[-1]).astype(np.uint8)
    lbl_set = np.unique(flat_lbls_0, axis=0)

    return lbl_set

def texturize(onehot):
    flat_onehot = onehot.reshape(-1, onehot.shape[-1])
    lbl_set = np.unique(flat_onehot, axis=0)

    count_lbls = [np.all(flat_onehot == ll, axis=-1).sum() for ll in lbl_set]
    object_id = np.argsort(count_lbls)[::-1][1]

    hidxs = []
    for h in range(onehot.shape[0]):
        appears = np.any(onehot[h, :, 1:] == 1)
        if appears:    
            hidxs.append(h)

    nstripes = min(10, len(hidxs))

    out = np.zeros((*onehot.shape[:2], nstripes+1))
    out[:, :, 0] = 1

    for i, h in enumerate(hidxs):
        cidx = int(i // (len(hidxs) / nstripes))
        w = np.any(onehot[h, :, 1:] == 1, axis=-1)
        out[h][w] = 0
        out[h][w, cidx+1] = 1

    return out


class YoutubeDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set, json_file, args):

        self.filelist = args.filelist
        self.imgSize = args.imgSize
        self.videoLen = args.videoLen

        self.texture = args.texture
        self.round = args.round

        self._voc_init(root, image_set, json_file)
        self.ignore_index = 255
        

    def get_onehot_lbl(self, lbl_path):
        name = '/' + '/'.join(lbl_path.split('.')[:-1]) + '_onehot.npy'
        if os.path.exists(name):
            return np.load(name)
        else:
            return None
    

    def make_paths(self, folder_path):
        frame_num =len(folder_path) + self.videoLen
        I_out = []

        for i in range(frame_num):
            i = max(0, i - self.videoLen)
            img_path = folder_path[i]
            
            I_out.append(img_path)

        return I_out


    def __getitem__(self, index):

        
        start_path = sequence_to_string(unpack_sequence(self.image_v_s, self.image_o_s, index))
        end_path = sequence_to_string(unpack_sequence(self.image_v_e, self.image_o_e, index))
        start_path_,end_path_,shot_idx = self.both_candid[index]

        class_name = start_path.split('/')[-4]
        ytid = start_path.split('/')[-3]

        start_idx = int(start_path.split('/')[-1].split('_save_')[-1].split('.')[-2])
        end_idx = int(end_path.split('/')[-1].split('_save_')[-1].split('.')[-2])

        meta = {}
        meta['class_name'] = class_name
        meta['ytid'     ]=ytid
        meta['start_idx']=start_idx
        meta['end_idx'  ]=end_idx
        meta['shot_idx' ]=shot_idx

        return meta
    
    def __len__(self):
        return self.image_len

    def _voc_init(self, root, image_set, json_file):
        
        start_candidates = []
        end_candidates = []
        both_candidates = []

        # target_ytid = "--J5HChVy1I"
        # target_category = "airplane"

        print("loading data path..")
        with open(json_file) as jfile:
            data = json.load(jfile)
            self.json_data = data
            for cls_name in data.keys():
                # print(cls_name)
                # if cls_name != target_category:
                #     continue
                for scls_name in data[cls_name].keys():
                    # print(scls_name)
                    if len(data[cls_name][scls_name]) == 0:
                        continue
                    # elif (scls_name != target_ytid):
                    #     continue
                    else:
                        for shot_idx in sorted(data[cls_name][scls_name].keys()):
                            # print(cls_name, scls_name, shot_idx)
                            # print(cls_name, scls_name, shot_idx)
                            if shot_idx == 'skip_cnt':
                                continue
                            
                            prefix = os.path.join(root, image_set, cls_name, scls_name, "frames")
                            
                            start_frame, end_frame = data[cls_name][scls_name][shot_idx]
                            skip_cnt = data[cls_name][scls_name]["skip_cnt"]

                            start_idx = round((start_frame)/skip_cnt)
                            end_idx = round((end_frame-1)/skip_cnt)

                            # end_idx = min(start_idx+100, end_idx)

                            if start_idx == end_idx:
                                continue

                            candid_sufix_later = "_save_"+"{0:010d}".format(start_idx)
                            candid_sufix_former ="real_"+"{0:010d}".format(start_idx*skip_cnt)
                            pic_name = candid_sufix_former+candid_sufix_later+".png"
                            start_img = os.path.join(prefix, pic_name)

                            if os.path.isfile(start_img) == False:
                                print(cls_name, scls_name, shot_idx, start_idx, end_idx)
                                print("missing file -> ", start_img)
                            else:
                                start_candidates.append(string_to_sequence(start_img))
                            
                            candid_sufix_later_end = "_save_"+"{0:010d}".format(end_idx)
                            candid_sufix_former_end ="real_"+"{0:010d}".format(end_idx*skip_cnt)
                            pic_name_end = candid_sufix_former_end+candid_sufix_later_end+".png"
                            end_img = os.path.join(prefix, pic_name_end)

                            if os.path.isfile(end_img) == False:
                                print(cls_name, scls_name, shot_idx, start_frame, end_frame-1, start_idx, end_idx)
                                print("missing file -> ", end_img)
                            else:
                                end_candidates.append(string_to_sequence(end_img))
                            
                            if (os.path.isfile(start_img) == True) and (os.path.isfile(end_img) == True):
                                both_candidates.append((start_img, end_img, shot_idx))
                            
                            # break

        self.image_len = len(start_candidates)
        self.image_v_s, self.image_o_s = pack_sequences(start_candidates)    
        self.image_v_e, self.image_o_e = pack_sequences(end_candidates)    
        self.both_candid = both_candidates
        print("Done.")
        

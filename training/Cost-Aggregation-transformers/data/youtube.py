from __future__ import print_function, absolute_import

import os

import numpy as np
import torch
import json
from os.path import join as pjn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import random
try:
    from data.keypoint_to_flow import KeypointToFlow
    import data.transforms as T
except:
    from keypoint_to_flow import KeypointToFlow
    import transforms as T

import cv2
import albumentations as A

class YoutubeDataset(torch.utils.data.Dataset):
    def __init__(self, image_set, json_file, pl_path, feature_size):

        self._yt_init(image_set, json_file, pl_path)
        self.imside = 256
        # self.img_transform = transforms.Compose(
        #     [
        #         transforms.Resize((self.imside, self.imside)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225])
        #     ]
        # )
        self.max_pts = 40

        random_transform = transforms.RandomApply([
            transforms.ColorJitter(0.2,0.2,0.2,0.2)
        ], p=0.2)

        self.img_transform_src = transforms.Compose([
            # transforms.RandomPosterize(0.2),
            transforms.RandomEqualize(0.2), 
            transforms.RandomSolarize(128, 0.2),
            transforms.RandomGrayscale(0.2),
            random_transform
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

        self.img_transform_tgt = transforms.Compose([
            # transforms.RandomPosterize(0.2),
            transforms.RandomEqualize(0.2), 
            transforms.RandomSolarize(128, 0.2),
            transforms.RandomGrayscale(0.2),
            random_transform
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])



        self.transform = T.Compose([
            # T.RandomHorizontalFlip(0.5),
            # T.RandomVerticalFlip(0.5),
            T.RandomPerspective(0.7, 0.5)
        ])

        self.normalize = transforms.Compose([
            transforms.Resize((self.imside, self.imside)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

        self.kps_to_flow = KeypointToFlow(receptive_field_size=35, jsz=256//feature_size, feat_size=feature_size, img_size=self.imside)

        

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

    def get_data(self, index):
        src_path_pred = self.src_path_pred_list[index]
        tgt_path_pred = self.tgt_path_pred_list[index]

        src_path_img = self.src_path_img_list[index]
        tgt_path_img = self.tgt_path_img_list[index]
        
        src_pred = torch.from_numpy(np.load(src_path_pred)).unsqueeze(0)
        tgt_pred = torch.from_numpy(np.load(tgt_path_pred)).unsqueeze(0)

        src_img_pil = self.get_image(src_path_img)
        src_img_tr = self.img_transform_src(src_img_pil)
        
        tgt_img_pil = self.get_image(tgt_path_img)
        tgt_img_tr = self.img_transform_tgt(tgt_img_pil)

        src_img_tr, src_pred = self.transform(src_img_tr, src_pred)
        src_pred = src_pred.squeeze(0).long()
        src_img = self.normalize(src_img_tr)

        tgt_img_tr, tgt_pred = self.transform(tgt_img_tr, tgt_pred)
        tgt_pred = tgt_pred.squeeze(0).long()
        tgt_img = self.normalize(tgt_img_tr)

        target_idx = np.intersect1d(np.unique(tgt_pred), np.unique(src_pred))
        target_idx = target_idx[target_idx!=-99]

        return src_img, src_pred, tgt_img, tgt_pred, target_idx

    def get_data_heuristic(self, index):
        
        src_pred, tgt_pred = self.apply_heuristics(index)

        src_pred = torch.from_numpy(src_pred).unsqueeze(0)
        tgt_pred = torch.from_numpy(tgt_pred).unsqueeze(0)

        src_path_img = self.src_path_img_list[index]
        tgt_path_img = self.tgt_path_img_list[index]
        
        src_img_pil = self.get_image(src_path_img)
        src_img_tr = self.img_transform_src(src_img_pil)
        
        tgt_img_pil = self.get_image(tgt_path_img)
        tgt_img_tr = self.img_transform_tgt(tgt_img_pil)

        src_img_tr, src_pred = self.transform(src_img_tr, src_pred)
        src_pred = src_pred.squeeze(0).long()
        src_img = self.normalize(src_img_tr)

        tgt_img_tr, tgt_pred = self.transform(tgt_img_tr, tgt_pred)
        tgt_pred = tgt_pred.squeeze(0).long()
        tgt_img = self.normalize(tgt_img_tr)

        target_idx = np.intersect1d(np.unique(tgt_pred), np.unique(src_pred))
        target_idx = target_idx[target_idx!=-99]

        return src_img, src_pred, tgt_img, tgt_pred, target_idx

    def __getitem__(self, index):

        while True:
            src_img, src_pred, tgt_img, tgt_pred, target_idx = self.get_data(index)
            if len(target_idx) > 0:
                break
        if len(target_idx) > self.max_pts:
            target_idx = np.random.choice(target_idx, size=self.max_pts, replace=False)
        
        target_idx = torch.from_numpy(target_idx)
        src_kps = []
        tgt_kps = []
        for st in target_idx:
                k = torch.nonzero(src_pred == st, as_tuple=False)
                base_y, base_x = k[0]
                base_y = base_y.item()
                base_x = base_x.item()

                src_kps.append([base_x, base_y])

                tgt_candid = torch.nonzero(tgt_pred == st, as_tuple=False)
                baset_y, baset_x = tgt_candid[0]
                baset_y = baset_y.item()
                baset_x = baset_x.item()

                tgt_kps.append([baset_x, baset_y])

                src_kps_, num_pts = self.get_points(torch.Tensor(src_kps).t().float())
                tgt_kps_, _ = self.get_points(torch.Tensor(tgt_kps).t().float())

                src_kps_raw = torch.Tensor(src_kps).t().float()
                tgt_kps_raw = torch.Tensor(tgt_kps).t().float()
                n_pts = torch.tensor(num_pts)
        if random.uniform(0, 1) < 0.5:
            batch = {}
            batch['src_kps'] = tgt_kps_
            batch['trg_kps'] = src_kps_
            batch['src_kps_raw'] = tgt_kps_raw
            batch['tgt_kps_raw'] = src_kps_raw
            batch['n_pts'] = n_pts
            batch['src_img'] = tgt_img
            batch['tgt_img'] = src_img
            batch['src_pred'] = tgt_pred
            batch['tgt_pred'] = src_pred
            flow = self.kps_to_flow(batch)
            batch['flow'] = flow
            return (
                tgt_img, src_img, tgt_pred, src_pred, 
                flow
            )
        else:
            batch = {}
            batch['src_kps'] = src_kps_
            batch['trg_kps'] = tgt_kps_
            batch['src_kps_raw'] = src_kps_raw
            batch['tgt_kps_raw'] = tgt_kps_raw 
            batch['n_pts'] = n_pts
            batch['src_img'] = src_img
            batch['tgt_img'] = tgt_img
            batch['src_pred'] = src_pred
            batch['tgt_pred'] = tgt_pred
            flow = self.kps_to_flow(batch)
            # print(index, src_img.shape, tgt_img.shape, src_pred.shape, tgt_pred.shape, 
            #     flow.shape)
            return (
                src_img, tgt_img, src_pred, tgt_pred, 
                flow
            )



    def apply_heuristics(self, index):
        src_path_img = self.src_path_img_list[index]
        tgt_path_img = self.tgt_path_img_list[index]

        src_img = cv2.resize(cv2.imread(os.path.join(src_path_img)), (256, 256))
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

        tgt_img = cv2.resize(cv2.imread(os.path.join(tgt_path_img)), (256, 256))
        tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(src_img, None)
        kp2, des2 = orb.detectAndCompute(tgt_img, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            good_matches = matches[:1000]
        except:
            good_matches = []
    

        matched_points1 = []
        matched_points2 = []
        for match in good_matches:
            # queryIdx: 첫 번째 이미지에서의 키포인트 인덱스
            # trainIdx: 두 번째 이미지에서의 키포인트 인덱스
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            matched_points1.append((int(pt1[0]), int(pt1[1])))
            matched_points2.append((int(pt2[0]), int(pt2[1])))
        
        src_pred = np.full((256, 256), -99)
        tgt_pred = np.full((256, 256), -99)


        for idx, ((x1, y1), (x2, y2)) in enumerate(zip(matched_points1, matched_points2)):
            src_pred[y1, x1] = idx
            tgt_pred[y2, x2] = idx            
            
        return src_pred, tgt_pred

    def __len__(self):
        return self.len

    def _yt_init(self, image_set, json_file, pl_path):
        
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
        print(json_file)
        with open(json_file) as jfile:
            data = json.load(jfile)
        
        try:
            with open('./file_list.json') as f:
                data_ = json.load(f)
            self.src_path_pred_list = data_['src_path_pred_list']
            self.tgt_path_pred_list = data_['tgt_path_pred_lis'] 
            self.src_path_img_list  = data_['src_path_img_list'] 
            self.tgt_path_img_list  = data_['tgt_path_img_list'] 

            self.len = len(self.src_path_img_list)
        except:
            save_dict = {}
            
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
                            print(s_class, s_ytid, s_shot)
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
            self.src_path_pred_list = src_path_pred_list
            self.tgt_path_pred_list = tgt_path_pred_list
            self.src_path_img_list = src_path_img_list
            self.tgt_path_img_list = tgt_path_img_list
            
            save_dict['src_path_pred_list'] = src_path_pred_list
            save_dict['tgt_path_pred_lis'] = tgt_path_pred_list
            save_dict['src_path_img_list'] = src_path_img_list
            save_dict['tgt_path_img_list'] =  tgt_path_img_list
    
            with open('./file_list.json','w') as f:
                json.dump(save_dict, f, ensure_ascii=False, indent=4)
                
            self.len = len(self.src_path_img_list)
            print("Done.")
            print("# of yt videos : ", yt_video)
            print("# of yt shots : ", yt_shot)
            print("# of yt pairs : ", yt_pair)
            print("# of yt frames : ", yt_frame)
        # exit(1)
        

def imwrite(path, img):
    img = img*255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)
if __name__ == "__main__":
    colors = []
    for _ in range(512*512):
        color = np.random.choice(256, 3, replace=True)
        color = (int(color[0]), int(color[1]), int(color[2]))
        colors.append(color)

    our_size = 256

    youtube_dataset = YoutubeDataset(
        image_set="/data01/kinux98/backup/YoutubeCrawling/videos_thumbnail_new",
        json_file="./video_scene_parsing_new_mt.json",
        pl_path="/data01/kinux98/backup/kinux98/SCvideo/videowalk/code/results/",
    )

    idx = 888
    src_img, tgt_img, src_pred, tgt_pred, _, _, _, flow= youtube_dataset.__getitem__(idx)
    src_img = src_img.permute(1,2,0).numpy()
    tgt_img = tgt_img.permute(1,2,0).numpy()

    src_pred = src_pred.numpy()
    tgt_pred = tgt_pred.numpy()

    outpath = os.path.join("./test_fig", str(idx))

    print(src_img.shape, src_pred.shape)

    if True:
        direction = 'horizontal'
        intersect = np.intersect1d(np.unique(tgt_pred), np.unique(src_pred))
        if direction == 'vertical':
            source_image_ = (torch.from_numpy(src_img)).squeeze() # h x w x 3
            target_image_ = (torch.from_numpy(tgt_img)).squeeze() # h x w x 3
            concat_image = torch.vstack([source_image_,  target_image_]).numpy() # 2h x w x 3
        else:
            source_image_ = (torch.from_numpy(src_img)).squeeze() # h x w x 3
            target_image_ = (torch.from_numpy(tgt_img)).squeeze() # h x w x 3
            concat_image = torch.hstack([source_image_,  target_image_]).numpy() # h x 2w x 3
        
        concat_untaced = concat_image.copy()
        candid = torch.nonzero((torch.from_numpy(tgt_pred) != -99), as_tuple=False)
        np.random.seed(20170890)
        # elected_candidate = np.random.choice(intersect, min(50, len(candid)-1), replace=True)
        elected_candidate = np.random.choice(intersect, 10, replace=False)
        for ccnt, rnd_sampled_idx in enumerate(elected_candidate):
            k = torch.nonzero(torch.from_numpy(src_pred) == rnd_sampled_idx,as_tuple=False)
            base_y, base_x = k[0]
            base_y = base_y.item()
            base_x = base_x.item()
            base_y = int(base_y * 256/our_size)
            base_x = int(base_x * 256/our_size)
            corr_candid = torch.nonzero(torch.from_numpy(tgt_pred) == rnd_sampled_idx, as_tuple=False)
            # elected_candidate_corr = np.random.choice(len(corr_candid), min(10, len(corr_candid)-1), replace=False)
            elected_candidate_corr = np.random.choice(len(corr_candid), len(corr_candid), replace=False)
            for cccnt, rnd_sampled_idx_corr in enumerate(elected_candidate_corr):
                corr_coord = corr_candid[rnd_sampled_idx_corr]
                wrpd_y = corr_coord[0].item()
                wrpd_x = corr_coord[1].item()
                wrpd_y = int(wrpd_y * 256/our_size)
                wrpd_x = int(wrpd_x * 256/our_size)
                dest_coord = (base_x, base_y)
                source_coord = (wrpd_x+256, wrpd_y)
                # print(source_coord, dest_coord)
                cv2.line(concat_image, source_coord, dest_coord, colors[rnd_sampled_idx_corr], thickness=1, lineType=cv2.LINE_AA)
        path_ = os.path.join(outpath , "draw_line")
        if not os.path.isdir(path_):
            os.makedirs(path_)
        final_path = os.path.join(path_, "img_"+str(direction)+str(0)+"_to_"+str(idx)+".jpg")
        print(final_path)
        imwrite(final_path, concat_image)
        imwrite(os.path.join(outpath , "origin", "img_"+str(direction)+str(0)+"_to_"+str(idx)+".jpg"), concat_untaced)
                        
        
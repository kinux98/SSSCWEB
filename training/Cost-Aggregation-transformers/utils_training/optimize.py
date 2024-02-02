import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.keypoint_to_flow import KeypointToFlowCuda
from utils_training.utils import flow2kps, flow2flow
from utils_training.evaluation import Evaluator
import wandb
import gc
from torch.nn.functional import interpolate
import cv2

r'''
    loss function implementation from GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
def un_normalize(img, mean=imagenet_mean, std=imagenet_std):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img*255

def EPE(input_flow, target_flow, sparse=True, mean=True, sum=False):

    # if torch.isnan(target_flow).any():
    #     print("NAN in targetflow")
    # if torch.isnan(input_flow).any():
    #     print("NAN in inputflow")
    # if torch.isnan((target_flow-input_flow)).any():
    #     print("NAN in t-i")
    EPE_map = torch.norm(target_flow-input_flow, 2, 1)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)
        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/torch.sum(~mask)

def update_teacher(net, t_net):
    _contrast_momentum = 0.995
    for mean_param, param in zip(t_net.parameters(), net.parameters()):
            mean_param.data.mul_(_contrast_momentum).add_(1 - _contrast_momentum, param.data)

def calc_contrastive(net, t_net, src_img, tgt_img, src_pred, tgt_pred, device):
    src_feats, tgt_feats = net(
                tgt_img.to(device), 
                src_img.to(device),
                get_feat=True #, get_corr=True
            )
    # corr_s = corr_s.view(corr_s.size(0), 1, 16*16, 16*16)
    # corr_s = interpolate(corr_s, size=(4096, 4096), mode='bilinear', align_corners=True).squeeze()
    # corr_s = corr_s.reshape(corr_s.size(0),-1, 64, 64) 
    
    with torch.no_grad():
        if t_net != None:
            src_feats_t, tgt_feats_t, corr = t_net(
                tgt_img.to(device), 
                src_img.to(device),
                get_feat=True, get_corr=True
            )
        else:
            src_feats_t, tgt_feats_t, corr = net(
                tgt_img.to(device), 
                src_img.to(device),
                get_feat=True, get_corr=True
            )
        src_feats_t = src_feats_t.detach()
        tgt_feats_t = tgt_feats_t.detach()
        corr = corr.detach()
        # print("corr shape : ", corr.shape) # b x 256 x 256
        # exit(1)
        corr = corr.view(corr.size(0), 1, 16*16, 16*16)
        corr = interpolate(corr, size=(4096, 4096), mode='bilinear', align_corners=True).squeeze()
        corr = corr.reshape(corr.size(0),-1, 256, 256) 

    # corr shape :  torch.Size([40, 256, 256])
    b_loss = 0
    for sb in range(src_feats.size(0)): #
        src_sb = F.interpolate(src_feats[sb].unsqueeze(0), size=(256,256), mode='bilinear', align_corners=True).squeeze() # c x h x w

        tgt_sb = F.interpolate(tgt_feats[sb].unsqueeze(0), size=(256,256), mode='bilinear', align_corners=True).squeeze() # c x h x w

        src_sb_t = F.interpolate(src_feats_t[sb].unsqueeze(0), size=(256,256), mode='bilinear', align_corners=True).squeeze() # c x h x w
        tgt_sb_t = F.interpolate(tgt_feats_t[sb].unsqueeze(0), size=(256,256), mode='bilinear', align_corners=True).squeeze() # c x h x w

        src_pred_sb = src_pred[sb] # h x w
        tgt_pred_sb = tgt_pred[sb] # h x w

        target_idx = np.intersect1d(torch.unique(tgt_pred_sb).detach().cpu().numpy(), torch.unique(src_pred_sb).detach().cpu().numpy())
        target_idx = target_idx[target_idx!=-99]
        
        if len(target_idx) > 10: # 10
            target_idx = np.random.choice(target_idx, size=10, replace=False)
        # else:
        #     target_idx = np.unique(np.random.choice(target_idx, size=40, replace=True))
        target_idx = torch.from_numpy(target_idx).cuda()
        
        i_loss = 0

        c_loss = 0
        
        for idx in target_idx:

            corr_b = corr[sb, :, src_pred_sb == idx].squeeze() # 4096
            corr_b = F.interpolate(corr_b.unsqueeze(0).unsqueeze(0).unsqueeze(0), size=(1,256*256), mode='bilinear', align_corners=True).squeeze()
            masked = torch.zeros_like(corr_b).cuda()
            values, _ = torch.topk(corr_b, k=25) # 410 , 100
            masked[corr_b >= values[-1]] = 1
            masked = masked.reshape(256,256)
            masked[tgt_pred_sb == idx] = 0

            ## student 
            src_vec_pos = src_sb[:, src_pred_sb == idx].T # n x 128
            src_vec_pos = src_vec_pos/ src_vec_pos.norm(dim=1, keepdim=True).clamp(min=1e-8)
           
            tgt_vec_pos = tgt_sb[:, tgt_pred_sb == idx].T # n x 128
            tgt_vec_pos = tgt_vec_pos / tgt_vec_pos.norm(dim=1, keepdim=True).clamp(min=1e-8)
        
            pos_vec_cat = torch.cat([src_vec_pos, tgt_vec_pos], dim=0) # 2n x 128

            ## teacher 
            src_vec_pos_t = src_sb_t[:, src_pred_sb == idx].T # n x 128
            src_vec_pos_t = src_vec_pos_t/ src_vec_pos_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
           
            tgt_vec_pos_t = tgt_sb_t[:, tgt_pred_sb == idx].T # n x 128
            tgt_vec_pos_t = tgt_vec_pos_t / tgt_vec_pos_t.norm(dim=1, keepdim=True).clamp(min=1e-8)

            pos_pair_1 = torch.exp(torch.mm(src_vec_pos, tgt_vec_pos_t.T) / 0.5).sum().clamp(min=1e-8)
            pos_pair_2 = torch.exp(torch.mm(src_vec_pos_t, tgt_vec_pos.T) / 0.5).sum().clamp(min=1e-8)

            pos_pair = pos_pair_1 + pos_pair_2
            
            tgt_vec_neg_t = tgt_sb_t[: , masked==1] # 128 x n
            tgt_vec_neg_t = tgt_vec_neg_t / tgt_vec_neg_t.norm(dim=0, keepdim=True).clamp(min=1e-8)

            tgt_vec_neg = tgt_sb[:, masked==1] # 128 x n
            tgt_vec_neg = tgt_vec_neg / tgt_vec_neg.norm(dim=0, keepdim=True).clamp(min=1e-8)

            neg_pair_1 = torch.exp(torch.mm(src_vec_pos, tgt_vec_neg_t) / 0.5).sum().clamp(min=1e-8)
            neg_pair_2 = torch.exp(torch.mm(src_vec_pos_t, tgt_vec_neg) / 0.5).sum().clamp(min=1e-8)
            neg_pair = neg_pair_1 + neg_pair_2

            i_loss += -(torch.log(pos_pair / (neg_pair + pos_pair))) / (pos_vec_cat.size(0))

        b_loss += (i_loss / len(target_idx))
    
    b_loss = b_loss / src_feats.size(0)
    return b_loss * 0.1 

def calc_dmloss(net, da_disc, orig_src, orig_tgt, yt_src, yt_tgt, category_id, device, alpha):
    
    orig_feats = net(
                orig_tgt.to(device), 
                orig_src.to(device),
                get_feat_DANN=True
            )#.detach()    # b x 8 x h x w
    yt_feats = net(
                yt_tgt.to(device), 
                yt_src.to(device),
                get_feat_DANN=True
            )               # b x 8 x h x w
    # print("orig-feats : ", orig_feats.shape)
    src_pred = da_disc(orig_feats, alpha) # 2b
    src_label = torch.ones(src_pred.size(0)).cuda().unsqueeze(1)
    src_loss = F.binary_cross_entropy_with_logits(src_pred, src_label, reduction='mean') 
    
    tgt_pred = da_disc(yt_feats, alpha)
    tgt_label = torch.zeros(tgt_pred.size(0)).cuda().unsqueeze(1)
    tgt_loss = F.binary_cross_entropy_with_logits(tgt_pred, tgt_label, reduction='mean') 

    return src_loss, tgt_loss

def calc_semi_keypoint(net, flow_gt, src_img, tgt_img, device):
    
    pred_flow_yt_s = net(tgt_img.to(device), src_img.to(device))
    
    loss_by_crw = EPE(pred_flow_yt_s, flow_gt.to(device)) 

    return loss_by_crw.mean()

kps_to_flow = KeypointToFlowCuda(receptive_field_size=35, jsz=256//16, feat_size=16, img_size=256, is_cuda=True)

def check_cycle_consistency(net, src_img, tgt_img, n_pts, src_kps, trg_kps,  device):

    pred_tgt_to_src = net(tgt_img.to(device), src_img.to(device))
    pred_src_to_tgt = net(src_img.to(device), tgt_img.to(device))


    # estimated_kps_tgt_to_src = flow2kps(trg_kps.to(device), pred_tgt_to_src ,n_pts)
    # estimated_kps_src_to_tgt = flow2kps(src_kps.to(device), pred_src_to_tgt ,n_pts)

    estimated_flow_tgt_to_src = flow2flow(trg_kps.to(device), pred_tgt_to_src ,n_pts)
    estimated_flow_src_to_tgt = flow2flow(src_kps.to(device), pred_src_to_tgt ,n_pts)

    src_kps = src_kps.to(device)
    trg_kps = trg_kps.to(device)


    # print(src_kps[0])
    # print(estimated_kps_tgt_to_src[0].long())

    # print(trg_kps[0])
    # print(estimated_kps_src_to_tgt[0].long())

    # print(estimated_flow_tgt_to_src[0])
    # print(estimated_flow_src_to_tgt[0])

    # print((estimated_flow_tgt_to_src + estimated_flow_src_to_tgt)[0])

    pre_mask = (src_kps != -1)[:, 0].cuda()

    # exit()
    threshold = 4
    
    summ = torch.abs(estimated_flow_tgt_to_src + estimated_flow_src_to_tgt)
    x_summ = (summ[:, 0] < threshold).cuda()
    y_summ = (summ[:, 1] < threshold).cuda()

    mask = x_summ & y_summ

    mask = (pre_mask & mask).unsqueeze(1)
    mask = torch.cat([mask, mask], dim=1)

    # print(mask[0])

    src_kps[mask == False] = -1
    trg_kps[mask == False] = -1

    new_flow = []
    mask = []

    for b in range(src_kps.shape[0]):
        batch = {}

        # print(src_kps.shape)
        # print(src_kps[b])
        # print(trg_kps[b])

        n_pts = len(src_kps[b][0][src_kps[b][0] == -1])

        pad = torch.zeros(n_pts).fill_(-1.).cuda().unsqueeze(1)
        pad = torch.cat([pad, pad], dim=1).permute(1,0)

        src_x = src_kps[b][0]
        src_x = src_x[src_x != -1]

        src_y = src_kps[b][1]
        src_y = src_y[src_y != -1]

        trg_x = trg_kps[b][0]
        trg_x = trg_x[trg_x != -1]
        
        trg_y = trg_kps[b][1]
        trg_y = trg_y[trg_y != -1]

        src = torch.stack((src_x, src_y))
        src = torch.hstack((src, pad))

        trg = torch.stack((trg_x, trg_y))
        trg = torch.hstack((trg, pad))

        # print(src)
        # print(trg)
        
        l = len(src_x[src_x!=-1])
        
        # exit()

        batch['src_kps'] = src
        batch['trg_kps'] = trg
        batch['n_pts']   = l

        flow = kps_to_flow(batch)
        new_flow.append(flow.unsqueeze(0))

    new_flow  = torch.cat(new_flow, dim=0)
    return new_flow, summ

unsup = False
semisup = True

semisup_epoch = 4

def train_epoch(net,
                t_net,
                is_sgood,
                scaler,
                da_disc,
                optimizer,
                train_loader,
                youtube_loader,
                device,
                epoch, args):
    
    
    net.train()
    if t_net is not None:
        t_net.train()   
    if (epoch > semisup_epoch) and (t_net is not None):
        net.module.load_state_dict(t_net.module.state_dict())
        # t_net.module.load_state_dict(net.module.state_dict())
        # pass
    elif (epoch == semisup_epoch) and (t_net is not None):
        t_net.module.load_state_dict(net.module.state_dict())

    running_total_loss = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    iter_per_epoch = len(train_loader)


    for i, mini_batch in pbar:
        optimizer.zero_grad()

        total_loss = 0
        ( 
            src_img, tgt_img, src_pred, tgt_pred,
            flow_gt_yt
        ) = next(youtube_loader)

        
        if args.run_sb == True:
            flow_gt = mini_batch['flow'].to(device)
            pred_flow = net(mini_batch['trg_img'].to(device),
                         mini_batch['src_img'].to(device))
            
            Loss = EPE(pred_flow, flow_gt) 
            if torch.isnan(Loss).any():
                Loss = 0
            else:
                wandb.log({
                "loss/train loss" : Loss.item(),
                }, commit=False)
                total_loss += Loss
            
        if args.run_yt == True:
            pred_flow_yt_s = net(tgt_img.to(device), src_img.to(device))
            loss_by_crw = EPE(pred_flow_yt_s, flow_gt_yt.to(device))

            
            if (args.run_sb == True) :#or (args.run_contra == True):
                loss_by_crw = loss_by_crw * 0.1
            elif (epoch >= semisup_epoch) and (args.run_contra == True):
                loss_by_crw = loss_by_crw * 0.01

            if torch.isnan(loss_by_crw).any():
                loss_by_crw = 0
            else:
                wandb.log({
                    "loss/semi_by_crw_loss" : loss_by_crw.item(),
                }, commit=False)
                total_loss += loss_by_crw
 

        if (da_disc is not None):# and (epoch > semisup_epoch):
            start_steps = epoch * iter_per_epoch
            total_steps = 1000 * iter_per_epoch
            p = float(i + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # alpha = 1.
            src_loss, tgt_loss = calc_dmloss(net, da_disc, mini_batch['src_img'], mini_batch['trg_img'], src_img, tgt_img, mini_batch['category_id'], device, alpha)
            if torch.isnan(src_loss).any() or torch.isnan(tgt_loss).any():
                src_loss= tgt_loss = 0
            else:
                wandb.log({
                "loss/dm_src_loss" : src_loss.item(),
                "loss/dm_tgt_loss" : tgt_loss.item(),
                "stat/alpha" : alpha
                }, commit=False)
                total_loss += (src_loss + tgt_loss)*0.5
            
        if total_loss == 0:
            continue
        
        scaler.scale(total_loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.001)

        if da_disc is not None:
            torch.nn.utils.clip_grad_norm_(da_disc.parameters(), 0.001)

        scaler.step(optimizer)
        scaler.update()

        if (epoch >= semisup_epoch) and (t_net is not None):
            update_teacher(net, t_net)

        running_total_loss += total_loss.item()
        wandb.log({
            "loss/total loss" : total_loss.item()
        })

        # n_iter += 1
        pbar.set_description(
                'training: R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), total_loss.item()))


    running_total_loss /= len(train_loader)
    return running_total_loss


def validate_epoch(net,
                   val_loader,
                   device,
                   epoch):
    net.eval()
    running_total_loss = 0

    # class_dict={} 

    # class_dict['aeroplane'] = []
    # class_dict['bicycle'] = []
    # class_dict['bird'] = []
    # class_dict['boat'] = []
    # class_dict['bottle'] = []
    # class_dict['bus'] = []
    # class_dict['car'] = []
    # class_dict['cat'] = []
    # class_dict['chair'] = []
    # class_dict['cow'] = []
    # class_dict['dog'] = []
    # class_dict['horse'] = []
    # class_dict['motorbike'] = []
    # class_dict['person'] = []
    # class_dict['pottedplant'] = []
    # class_dict['sheep'] = []
    # class_dict['train'] = []
    # class_dict['tvmonitor'] = []


    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        pck_array = []
        for i, mini_batch in pbar:
            flow_gt = mini_batch['flow'].to(device)
            pred_flow = net(mini_batch['trg_img'].to(device),
                            mini_batch['src_img'].to(device))

            estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))

            eval_result = Evaluator.eval_kps_transfer(estimated_kps.cpu(), mini_batch)
            
            Loss = EPE(pred_flow, flow_gt) 

            pck_array += eval_result['pck']

            # class_dict[mini_batch['category'][0]].append(eval_result['pck'])

            running_total_loss += Loss.item()
            pbar.set_description(
                ' validation R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), Loss.item()))
        mean_pck = sum(pck_array) / len(pck_array)

    return running_total_loss / len(val_loader), mean_pck#, class_dict
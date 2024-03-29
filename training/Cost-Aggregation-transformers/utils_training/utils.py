import re
import os
import shutil

import torch
import torch.nn.functional as F
import numpy as np

r'''
    source code from GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''
def save_checkpoint(state, is_best, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth'))

r'''
    source code from GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''
def load_checkpoint(model, da_disc, scaler, optimizer, scheduler, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    best_val=-1
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        # start_epoch = checkpoint['epoch']

        def load_state_dict(model, pretrain):
            model_dict = {}
            state_dict = model.state_dict()
            for k, v in pretrain.items():
                if k in state_dict and "feature_extraction" in k:
                    print(k)
                    model_dict[k] = v
            state_dict.update(model_dict)
            model.load_state_dict(state_dict)
        
        # try:
            # model.load_state_dict(checkpoint['state_dict'])
        # except:
        
        load_state_dict(model, checkpoint['state_dict'])
        # try:
        #     load_state_dict(da_disc, checkpoint['da_disc'])
        # except:
        #     print("Cannot load da_disc")
        
        # try:
        #     scaler.load_state_dict(checkpoint['scaler'])
        # except:
        #     print("Cannot load scaler")

        # try:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        # except:
        #     print("Cannot load optimizer.")

        # scheduler.load_state_dict(checkpoint['scheduler'])

        # try:
        #     best_val=checkpoint['best_loss']
        # except:
        #     best_val=-1
        # start_epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, da_disc, scaler, optimizer, scheduler, start_epoch, best_val 


# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def log_args(args):
    r"""Log program arguments"""
    print('\n+================================================+')
    for arg_key in args.__dict__:
        print('| %20s: %-24s |' % (arg_key, str(args.__dict__[arg_key])))
    print('+================================================+\n')


def parse_list(list_str):
    r"""Parse given list (string -> int)"""
    return list(map(int, re.findall(r'\d+', list_str)))


def where(predicate):
    r"""Predicate must be a condition on nd-tensor"""
    matching_indices = predicate.nonzero()
    if len(matching_indices) != 0:
        matching_indices = matching_indices.t().squeeze(0)
    return matching_indices


def flow2kps(trg_kps, flow, n_pts, upsample_size=(256, 256)):
    _, _, h, w = flow.size()
    flow = F.interpolate(flow, upsample_size, mode='bilinear') * (upsample_size[0] / h)
    
    src_kps = []
    for trg_kps, flow, n_pts in zip(trg_kps.long(), flow, n_pts):
        size = trg_kps.size(1)

        kp = torch.clamp(trg_kps.narrow_copy(1, 0, n_pts), 0, upsample_size[0] - 1)
        # print(kp)
        estimated_kps = kp + flow[:, kp[1, :], kp[0, :]]
        # print(estimated_kps)
        # exit()
        estimated_kps = torch.cat((estimated_kps, torch.ones(2, size - n_pts.item()).cuda() * -1), dim=1)
        src_kps.append(estimated_kps)

    return torch.stack(src_kps)


def flow2flow(trg_kps, flow, n_pts, upsample_size=(256, 256)):
    _, _, h, w = flow.size()
    flow = F.interpolate(flow, upsample_size, mode='bilinear') * (upsample_size[0] / h)
    
    src_kps = []
    for trg_kps, flow, n_pts in zip(trg_kps.long(), flow, n_pts):
        size = trg_kps.size(1)
        # print(trg_kps)
        kp = torch.clamp(trg_kps.narrow_copy(1, 0, n_pts), 0, upsample_size[0] - 1)
        # print(kp)
        estimated_kps = flow[:, kp[1, :], kp[0, :]]
        # print(estimated_kps)
        # exit()
        estimated_kps = torch.cat((estimated_kps, torch.ones(2, size - n_pts.item()).cuda() * -1), dim=1)
        src_kps.append(estimated_kps)

    return torch.stack(src_kps)
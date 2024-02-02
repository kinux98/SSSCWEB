import os
import torch
import numpy as np
from matplotlib import cm
import cv2

import torch.nn.functional as F
import imageio
import sys

def vis_pose(oriImg, points):

    pa = np.zeros(15)
    pa[2] = 0
    pa[12] = 8
    pa[8] = 4
    pa[4] = 0
    pa[11] = 7
    pa[7] = 3
    pa[3] = 0
    pa[0] = 1
    pa[14] = 10
    pa[10] = 6
    pa[6] = 1
    pa[13] = 9
    pa[9] = 5
    pa[5] = 1

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170,0,255],[255,0,255]]
    canvas = oriImg
    stickwidth = 4
    x = points[0, :]
    y = points[1, :]

    for n in range(len(x)):
        pair_id = int(pa[n])

        x1 = int(x[pair_id])
        y1 = int(y[pair_id])
        x2 = int(x[n])
        y2 = int(y[n])

        if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
            cv2.line(canvas, (x1, y1), (x2, y2), colors[n], 8)

    return canvas

def hard_prop(pred):
    pred_max = pred.max(axis=0)[0]
    pred[pred <  pred_max] = 0
    pred[pred >= pred_max] = 1
    pred /= pred.sum(0)[None]
    return pred

def process_pose(pred, lbl_set, topk=3):
    # generate the coordinates:
    pred = pred[..., 1:]
    flatlbls = pred.flatten(0,1)
    topk = min(flatlbls.shape[0], topk)
    
    vals, ids = torch.topk(flatlbls, k=topk, dim=0)
    vals /= vals.sum(0)[None]
    xx, yy = ids % pred.shape[1], ids // pred.shape[1]

    current_coord = torch.stack([(xx * vals).sum(0), (yy * vals).sum(0)], dim=0)
    current_coord[:, flatlbls.sum(0) == 0] = -1

    pred_val_sharp = np.zeros((*pred.shape[:2], 3))

    for t in range(len(lbl_set) - 1):
        x = int(current_coord[0, t])
        y = int(current_coord[1, t])

        if x >=0 and y >= 0:
            pred_val_sharp[y, x, :] = lbl_set[t + 1]

    return current_coord.cpu(), pred_val_sharp


def dump_predictions(pred, lbl_set, img, prefix):
    '''
    Save:
        1. Predicted labels for evaluation
        2. Label heatmaps for visualization
    '''
    sz = img.shape[:-1]

    # Upsample predicted soft label maps
    # pred_dist = pred.copy()
    pred_dist = cv2.resize(pred, sz[::-1])[:]
    
    # Argmax to get the hard label for index
    pred_lbl = np.argmax(pred_dist, axis=-1)
    pred_lbl = np.array(lbl_set, dtype=np.int32)[pred_lbl]      
    img_with_label = np.float32(img) * 0.5 + np.float32(pred_lbl) * 0.5

    # Visualize label distribution for object 1 (debugging/analysis)
    pred_soft = pred_dist[..., 1]
    pred_soft = cv2.resize(pred_soft, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_soft = cm.jet(pred_soft)[..., :3] * 255.0
    img_with_heatmap1 =  np.float32(img) * 0.5 + np.float32(pred_soft) * 0.5

    # Save blend image for visualization
    imageio.imwrite('%s_blend.jpg' % prefix, np.uint8(img_with_label))

    if prefix[-4] != '.':  # Super HACK-y
        imname2 = prefix + '_mask.png'
        # skimage.io.imsave(imname2, np.uint8(pred_val))
    else:
        imname2 = prefix.replace('jpg','png')
        # pred_val = np.uint8(pred_val)
        # skimage.io.imsave(imname2.replace('jpg','png'), pred_val)

    # Save predicted labels for evaluation
    imageio.imwrite(imname2, np.uint8(pred_lbl))

    return img_with_label, pred_lbl, img_with_heatmap1

def dump_predictions_yt(pred, prefix, t):
    '''
    Save:
        1. Predicted labels for evaluation
        2. Label heatmaps for visualization
    '''

    if len(torch.unique(pred)) == 1:
        pass
    else:
        pred = pred.cpu().numpy()
        # pred_ = cv2.resize(pred, (512, 512), interpolation=cv2.INTER_NEAREST)
        imname1 = os.path.join(prefix,"pred", t + '_pred.npy')  
        np.save(imname1, pred)

        # imname2 = os.path.join(prefix, "weights", t + '_weight.npy') 
        # np.save(imname2, weight)

        # imname3 = os.path.join(prefix, "coord",  t + '_coord.npy') 
        # np.save(imname3, coord)


    

# class Propagation:
#     def __init__(self, net, )

def context_index_bank(n_context, long_mem, N):
    '''
    Construct bank of source frames indices, for each target frame
    '''
    ll = []   # "long term" context (i.e. first frame)
    for t in long_mem:
        assert 0 <= t < N, 'context frame out of bounds'
        idx = torch.zeros(N, 1).long()
        if t > 0:
            idx += t + (n_context+1)
            idx[:n_context+t+1] = 0
        ll.append(idx)
    # "short" context    
    ss = [(torch.arange(n_context)[None].repeat(N, 1) +  torch.arange(N)[:, None])[:, :]]

    # return ll + ss
    return ss


def mem_efficient_batched_affinity(query, keys, mask, temperature, topk, long_mem, device):
    '''
    Mini-batched computation of affinity, for memory efficiency
    '''
    bsize, pbsize = 2, 100 #keys.shape[2] // 2
    Ws, Is = [], []

    for b in range(0, keys.shape[2], bsize):
        _k, _q = keys[:, :, b:b+bsize].to(device), query[:, :, b:b+bsize].to(device)
        w_s, i_s = [], []

        for pb in range(0, _k.shape[-1], pbsize):
            A = torch.einsum('ijklm,ijkn->iklmn', _k, _q[..., pb:pb+pbsize]) 
            A[0, :, len(long_mem):] += mask[..., pb:pb+pbsize].to(device)

            _, N, T, h1w1, hw = A.shape
            A = A.view(N, T*h1w1, hw)
            A /= temperature

            weights, ids = torch.topk(A, topk, dim=-2)
            weights = F.softmax(weights, dim=-2)
            
            w_s.append(weights.cpu())
            i_s.append(ids.cpu())

        weights = torch.cat(w_s, dim=-1)
        ids = torch.cat(i_s, dim=-1)
        Ws += [w for w in weights]
        Is += [ii for ii in ids]

    return Ws, Is

def batched_affinity_cossim(query, keys, topk, device):
    '''
    Mini-batched computation of affinity, for memory efficiency
    '''
    # print(query.shape)
    # print(keys.shape)
    query = query.squeeze() # d x hw
    keys = keys.squeeze() # d x hw

    query_norm = query / query.norm(dim=0, keepdim=True).clamp(min=1e-8)
    key_norm = keys / keys.norm(dim=0, keepdim=True).clamp(min=1e-8)

    sim = torch.mm(query_norm.T, key_norm)

    # weights, ids = torch.topk(sim, topk, dim=-1)
    weights, ids = torch.max(sim, dim=-1)

    return ids.unsqueeze(0)


def mem_efficient_batched_affinity_cossim(query, keys, topk, device):
    '''
    Mini-batched computation of affinity, for memory efficiency
    '''
    bsize, pbsize = 1, 100 #keys.shape[2] // 2
    # query / keys : d x hw

    query = query.squeeze()
    keys = keys.squeeze() # d x hw

    q = query / query.norm(dim=0, keepdim=True).clamp(min=1e-8)
    k = keys / keys.norm(dim=0, keepdim=True).clamp(min=1e-8)

    i_s = []
    for pb in range(0, q.shape[-1], pbsize):
        # print("_k, _q[..., pb:pb+pbsize] : ", _k.shape, _q[..., pb:pb+pbsize].shape)
        
        # _k = _k.squeeze() # 256 * 4096
        # _k_norm = _k / _k.norm(dim=0, keepdim=True).clamp(min=1e-8)
        # _q_t = _q[..., pb:pb+pbsize].squeeze().T # 100 * 256
        # _q_t_norm = _q_t / _q_t.norm(dim=1, keepdim=True).clamp(min=1e-8)

        _k = k.squeeze() # d * hw
        _q = q[:, pb:pb+pbsize].squeeze().T # 100 x d

        # print(_q_t_norm.shape, _k_norm.shape) # 100 x 256 , 256 x 4096

        sim = torch.mm(_q, _k) # 100 x hw
        _, ids = torch.max(sim, dim=-1)

        ids = ids.squeeze() # 100

        # print(sim.shape, weights.shape, ids.shape)
        
        i_s.append(ids)

        del _k, _q, sim, ids

    # print("i_s : ", len(i_s))
    ids_ = torch.cat(i_s)
    # print("ids_weights : ", ids_, ids_.shape)
    return ids_.unsqueeze(0)

def batched_affinity(query, keys, mask, temperature, topk, long_mem, device):
    '''
    Mini-batched computation of affinity, for memory efficiency
    (less aggressively mini-batched)
    '''
    bsize = 2
    Ws, Is = [], []
    for b in range(0, keys.shape[2], bsize):
        _k, _q = keys[:, :, b:b+bsize].to(device), query[:, :, b:b+bsize].to(device)
        w_s, i_s = [], []

        A = torch.einsum('ijklmn,ijkop->iklmnop', _k, _q) / temperature
        
        # Mask
        A[0, :, len(long_mem):] += mask.to(device)

        _, N, T, h1w1, hw = A.shape
        A = A.view(N, T*h1w1, hw)
        A /= temperature

        weights, ids = torch.topk(A, topk, dim=-2)
        weights = F.softmax(weights, dim=-2)
            
        Ws += [w for w in weights]
        Is += [ii for ii in ids]

    
    return Ws, Is

def infer_downscale(model):    
    out = model(torch.zeros(1, 10, 3, 320, 320).to(next(model.parameters()).device), just_feats=True)
    scale = out[1].shape[-2:]
    return 320 // np.array(scale)


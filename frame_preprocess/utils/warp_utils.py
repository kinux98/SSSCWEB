import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from enum import Enum

def get_corresponding_map(data):
    """

    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid

def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / max(W - 1, 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / max(H - 1, 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2

def flow_warp(x, flow12, pad='border', mode='bilinear', v_flow_r=False):
    B, _, H, W = x.size()
    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW
    v_grid = norm_grid(base_grid + flow12)  # BHW2
    # if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
    #     im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    # else:
        
    im1_recons = nn.functional.grid_sample(x.float(), v_grid.float(), mode=mode, padding_mode=pad, align_corners=True)

    if v_flow_r:
        return im1_recons, v_grid
    else:
        return im1_recons

def flow_warp_np(x, flow12, pad='border', mode='nearest', v_flow_r=False):
    
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float() # b x 3 x h x w
    flow12 = torch.from_numpy(flow12).float().unsqueeze(0).permute(0,3,1,2) # b x 2 x h x w

    B, _, H, W = x.size()
    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    # if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
    #     im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    # else:
        
    im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)

    if v_flow_r:
        return im1_recons, v_grid
    else:
        return im1_recons

# def flow_warp_kinux98(x, flow12):
    
#     x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float() # b x 3 x h x w
#     flow12 = torch.from_numpy(flow12).float().unsqueeze(0).permute(0,3,1,2) # b x 2 x h x w

#     B, _, H, W = x.size()
#     base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

#     v_grid = base_grid + flow12 # BHW2
#     v_grid_tmp = norm_grid(v_grid)
#     v_grid = v_grid.long()

#     outlier = torch.where((v_grid_tmp < -1 ) or (v_grid_tmp > 1), 0, 1).cuda()

#     flow12_ = flow12.permute(0,2,3,1) # b x h x w 2
#     im1_recons = torch.where(outlier == 1, ?, 0)
    
#     return im1_recons

def flow_warp2(x, flow, pad='border', mode='bilinear', v_flow_r=False):

    v_grid = norm_grid(flow.unsqueeze(0).float())  # BHW2

    im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)

    if v_flow_r:
        return im1_recons, v_grid
    else:
        return im1_recons

def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
        
    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()

def get_occu_mask_bidirection_np(flow12, flow21, scale=0.005, bias=0.25):
    
    flow12 = torch.from_numpy(flow12).float()
    flow21 = torch.from_numpy(flow21).float()

    flow12 = flow12.unsqueeze(0).permute(0,3,1,2)
    flow21 = flow21.unsqueeze(0).permute(0,3,1,2)

    # print(flow12.shape)

    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()

def get_occu_mask_backward(flow21, th=0.2):
    B, _, H, W = flow21.size()
    base_grid = mesh_grid(B, H, W).type_as(flow21)  # B2HW

    corr_map = get_corresponding_map(base_grid + flow21)  # BHW
    occu_mask = corr_map.clamp(min=0., max=1.) < th
    return occu_mask.float()

def flow_composition(flow_list, flow_list_rev, occ_type=2):
    '''
    flow1 + flow2 = flow3

    occ_type=0 : kinux98
    occ_type=1 : backward
    occ_type=2 : bidirection
    
    '''
    B, _, H, W = flow_list[0].size()

    vec_decomp_list = []

    def get_base(flow):
        base_grid = mesh_grid(B, H, W).type_as(flow)  # B2HW
        # v_grid_n = norm_grid(base_grid + flow)  # BHW2
        v_grid = base_grid + flow  # B2HW

        return torch.round(v_grid).long() # round to nearest integer
    
    for idx, single_flow in enumerate(flow_list):
        new_coord = get_base(single_flow)
        new_coord = new_coord.squeeze() # 2 x H x W
        vec_decomp_list.append(new_coord)
    
    base_flow_field = mesh_grid(B, H, W).type_as(vec_decomp_list[0]).squeeze() # 2 x H x W

    occ_list = []  
    bcc_list = []
    interm_flow_list = []
    for idx, single_decomp in enumerate(vec_decomp_list):

        if idx == 0:
            temp_flow_field = single_decomp[:, base_flow_field[1], base_flow_field[0]]
        else:
            tff_filtered = torch.where(occ_mask == 1, temp_flow_field, base_flow_field)
            temp_flow_field = single_decomp[:, tff_filtered[1], tff_filtered[0]]
        
        interm_flow_list.append(temp_flow_field)

        
        ## get occ mask
        x_field_flow = temp_flow_field[0]
        y_field_flow = temp_flow_field[1]

        occ_x = torch.where((x_field_flow >= 0) & (x_field_flow < W), 1, 0)
        occ_y = torch.where((y_field_flow >= 0) & (y_field_flow < H), 1, 0)

        occ_mask_default = (occ_x & occ_y).bool()
        
        if occ_type == 0:
            occ_mask_ = occ_mask_default
        elif occ_type == 1:
            occ_mask_ = 1 - get_occu_mask_backward(flow_list_rev[idx])
        elif occ_type == 2:
            occ_mask_ = 1 - get_occu_mask_bidirection(flow_list[idx], flow_list_rev[idx])

        if idx == 0:
            occ_mask = occ_mask_default
            # occ_mask = occ_mask_.bool().squeeze() & occ_mask_default
        else:
            # occ_mask = occ_mask & occ_mask_default
            occ_mask_ = occ_mask_.bool()
            occ_mask = occ_mask_.squeeze() & occ_mask_default
        
        ## new occ
        black_plane = torch.zeros((H, W)).cuda()
        new_flow = torch.where(occ_mask == 1, temp_flow_field, base_flow_field)
        black_plane[new_flow[1], new_flow[0]] = 1
        # black_plane = torch.where(occ_mask == 1, black_plane, torch.Tensor([0]).float().cuda())
        
        bcc_list.append(black_plane.bool())
        occ_list.append(occ_mask.bool())
        # temp_flow_field[:, occ_mask == False] = -999

    return interm_flow_list, occ_list, bcc_list


class Status(Enum):
    NONE = 0
    ONGOING = 1
    DONE = 2

def accumulate_flow(idx, track_dict, cflow, cflow_rev, occ_type=2):
    def get_base(flow):
        base_grid = mesh_grid(B, H, W).type_as(flow)  # B2HW
        # v_grid_n = norm_grid(base_grid + flow)  # BHW2
        v_grid = base_grid + flow  # B2HW

        return torch.round(v_grid).long() # round to nearest integer

    B, _, H, W = cflow.size()
    base_flow_field = mesh_grid(B, H, W).long().squeeze().cuda()

    for img_id in track_dict.keys():

        status = track_dict[img_id][2]

        new_coord = get_base(cflow).squeeze()
        
        if status == Status.NONE:
            temp_flow_field = new_coord[:, base_flow_field[1], base_flow_field[0]]
        elif status == Status.ONGOING:
            flow = track_dict[img_id][0][-1]
            occ_mask_prev = track_dict[img_id][1][-1]
            tff_filtered = torch.where(occ_mask_prev == 1, flow, base_flow_field)
            temp_flow_field = new_coord[:, tff_filtered[1], tff_filtered[0]]
        elif status == Status.DONE:
            continue
        
        x_field_flow = temp_flow_field[0]
        y_field_flow = temp_flow_field[1]

        occ_x = torch.where((x_field_flow >= 0) & (x_field_flow < W), 1, 0)
        occ_y = torch.where((y_field_flow >= 0) & (y_field_flow < H), 1, 0)

        occ_mask_default = occ_x & occ_y

        if occ_type == 0:
            occ_mask = occ_mask_default
        elif occ_type == 1:
            occ_mask = 1 - get_occu_mask_backward(cflow_rev)
        elif occ_type == 2:
            occ_mask = 1 - get_occu_mask_bidirection(cflow, cflow_rev)

        occ_mask = occ_mask.squeeze()

        if status == Status.ONGOING:
            occ_mask = occ_mask.bool() & occ_mask_default & occ_mask_prev
        else:
            occ_mask = occ_mask.bool() & occ_mask_default

        ratio = torch.count_nonzero(occ_mask) / torch.numel(occ_mask)

        if ratio > 0.6:
            flag = Status.ONGOING
        else:
            flag = Status.DONE

        prev_list = track_dict[img_id][0]
        prev_list.append(temp_flow_field)

        prev_list_occ = track_dict[img_id][1]
        prev_list_occ.append(occ_mask)

        track_dict[img_id] = (prev_list, prev_list_occ, flag)

def track_points(flow_list, flow_list_rev, occ_type=2):
    
    track_dict = {}

    for idx in range(len(flow_list)):
        if idx == 0:
            track_dict[idx] = ([], [], Status.NONE) # (accum_flow, occ_mask, is_done, count)
        accumulate_flow(idx, track_dict, flow_list[idx], flow_list_rev[idx], occ_type)

    return track_dict
    











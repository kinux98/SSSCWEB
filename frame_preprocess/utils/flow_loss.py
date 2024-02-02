import torch.nn as nn
import torch.nn.functional as F
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, TernaryLoss
from utils.warp_utils import flow_warp, mesh_grid
from utils.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward
import torch

class unFlowLoss_unitary(nn.modules.Module):
    def __init__(self):
        super(unFlowLoss_unitary, self).__init__()

    def loss_photomatric(self, im1_scaled, im1_recons):
        loss = []

        
        a = 0.1 * (im1_scaled - im1_recons).abs() 
        # if torch.isnan(a).any():
        #     print("NAN! w1")
        #     exit(-1)
        loss += [a]


        a = 0.5 * SSIM(im1_recons ,im1_scaled )
        # if torch.isnan(a).any():
        #     print("NAN! ssim")
        #     exit(-1)
        loss += [a]

        a = 0.4 * TernaryLoss(im1_recons,im1_scaled)
        # if torch.isnan(a).any():
        #     print("NAN! w_ternary")
        #     exit(-1)
        loss += [a]

        s_list=[]
        for idx, l in enumerate(loss):
            # print(idx,"loss.mean() : ", l.mean())
            s_list.append(l.mean())
        ret = sum(s_list) 

        if torch.isnan(ret).any():
            ret = torch.zeros(1).cuda()
        
        return ret

    def loss_smooth(self, flow, im1_scaled):
        func_smooth = smooth_grad_2nd
        
        loss = []
        loss += [func_smooth(flow, im1_scaled, 10)]
        return sum([l.mean() for l in loss])

    def forward(self, flow_fl, first_img, last_img):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """
        im1_origin = first_img
        im2_origin = last_img

        pyramid_smooth_losses = []
        pyramid_warp_losses = []
        s = 1.
    
        
        b, _, h, w = flow_fl.size()
        # resize images to match the size of layer
        im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
        im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')
        im2_recons = flow_warp(im1_scaled, flow_fl, pad='border')
        
     
        s = min(h, w)
        loss_smooth = 0
        loss_warp = 0

        loss_warp = self.loss_photomatric(im2_scaled, im2_recons)
        try:
            loss_smooth = self.loss_smooth(flow_fl / s, im2_scaled)
        except:
            pass
        pyramid_warp_losses.append(loss_warp)
        pyramid_smooth_losses.append(loss_smooth)

        pyramid_warp_losses = [l  for l in pyramid_warp_losses]
        pyramid_smooth_losses = [l  for l in pyramid_smooth_losses]

        warp_loss = sum(pyramid_warp_losses)
        if torch.isnan(warp_loss).any():
            warp_loss = torch.zeros(1).cuda()

        smooth_loss = 75. * sum(pyramid_smooth_losses)

        if torch.isnan(smooth_loss).any():
            smooth_loss = torch.zeros(1).cuda()

        total_loss = warp_loss + smooth_loss

        return total_loss, warp_loss, smooth_loss

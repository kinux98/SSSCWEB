
import torch
import numpy as np
import wandb
from utils.flow_utils import flow_to_image, resize_flow
import cv2
import torchvision.transforms.functional as F

from utils.warp_utils import get_occu_mask_bidirection

_mean = [0.5, 0.5, 0.5]
_std = [0.5, 0.5, 0.5]


def un_normalize(img, mean=_mean, std=_std):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

def dense_optical_flow(old_frame, new_frame):
    # Read the video and first frame
    # crate HSV & make Value a constant
    old_frame = np.transpose(old_frame, (1,2,0))
    old_frame = (255*(old_frame - np.min(old_frame))/np.ptp(old_frame)).astype(np.uint8)

    new_frame = np.transpose(new_frame, (1,2,0))
    new_frame = (255*(new_frame - np.min(new_frame))/np.ptp(new_frame)).astype(np.uint8)

    # Calculate Optical Flow
    flow = cv2.optflow.calcOpticalFlowDenseRLOF(old_frame, new_frame, None)
    return flow_to_image(flow)
        
def viz_input_image_pair(inputs):
    # inputs : b x 6 x h x w

    for sb in range(inputs.size(0)):
        image1 = inputs[sb][:3] # 3 x h x w
        image2 = inputs[sb][3:]

        image1 = un_normalize(image1).detach().cpu().numpy()
        image2 = un_normalize(image2).detach().cpu().numpy()

        wandb.log({
             "input_images/" + str(sb)+"_ImagePair_1" : [wandb.Image(np.transpose(image1, (1,2,0)))],
             "input_images/" + str(sb)+"_ImagePair_2" : [wandb.Image(np.transpose(image2, (1,2,0)))],
            }, commit=False) 

def viz_images(proj_imgs, targ_imgs):
    # inputs : T x 3 x h x w

    for idx, (p, t) in enumerate(zip(proj_imgs, targ_imgs)):
        p_ = un_normalize(p).detach().cpu().numpy()
        t_ = un_normalize(t).detach().cpu().numpy()

        wandb.log({
             "input_images/" +"_ImagePair_p_" + str(idx) : [wandb.Image(np.transpose(p_, (1,2,0)))],
             "input_images/" +"_ImagePair_t_" + str(idx) : [wandb.Image(np.transpose(t_, (1,2,0)))],
        }, commit=False) 


def put_optical_flow_arrows_on_image(image_, optical_flow_image, threshold=2.0, skip_amount=30):
    # Don't affect original image
    image = np.transpose(image_, (1,2,0))
    image = (255*(image - np.min(image))/np.ptp(image)).astype(np.uint8)
    image = image.copy()
    
    # Turn grayscale to rgb if needed
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=2)
    
    # Get start and end coordinates of the optical flow
    flow_start = np.stack(np.meshgrid(range(optical_flow_image.shape[1]), range(optical_flow_image.shape[0])), 2)
    flow_end = (optical_flow_image[flow_start[:,:,1],flow_start[:,:,0],:1]*3 + flow_start).astype(np.int32)
    

    # Threshold values
    norm = np.linalg.norm(flow_end - flow_start, axis=2)
    norm[norm < threshold] = 0
    
    # Draw all the nonzero values
    nz = np.nonzero(norm)
    for i in range(0, len(nz[0]), skip_amount):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(image,
                        pt1=tuple(flow_start[y,x]), 
                        pt2=tuple(flow_end[y,x]),
                        color=(0, 255, 0), 
                        thickness=1, 
                        tipLength=.1)
    return image

def viz_flow_image(inputs, flow_ori):
    # model_output : b x ?
    h, w = inputs.shape[2:]
    for sb in range(inputs.size(0)):
        flow_12 = flow_ori[sb]
        flow_12 = resize_flow(flow_12.unsqueeze(0), (h, w))

        np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])
        # vis_flow = flow_to_image(np_flow_12)
        
        image1 = inputs[sb][:3] # 3 x h x w
        image2 = inputs[sb][3:]

        image1 = un_normalize(image1).detach().cpu().numpy()
        image2 = un_normalize(image2).detach().cpu().numpy()

        vis_flow = flow_to_image(np_flow_12)

        arrows_flow = put_optical_flow_arrows_on_image(image1, np_flow_12)

        wandb.log({
            "pair" + str(sb)+"/_ImagePair_1" : [wandb.Image(np.transpose(image1, (1,2,0)))],
            "pair" + str(sb)+"/_ImagePair_2" : [wandb.Image(np.transpose(image2, (1,2,0)))],
            "pair" + str(sb)+"/_flow_image_net" : [wandb.Image(vis_flow)],
            "pair" + str(sb)+"/_arrow_flow_image_net" : [wandb.Image(arrows_flow)],
            }, commit=False) 


def viz_flow_image2(align_pred12, align_pred21, flow12, flow21, simg, timg):
    # align_pred, simg, timg : b x 3 x h x w
    # flow : b x h x w x 2
    b, _, h, w = simg.shape

    for sb in range(b):
        flow_12 = flow12[sb]
        flow_21 = flow21[sb]
        # flow_12 = resize_flow(flow_12.unsqueeze(0), (h, w))

        np_flow_12 = flow_12.detach().cpu().numpy()
        np_flow_21 = flow_21.detach().cpu().numpy()
        # vis_flow = flow_to_image(np_flow_12)
        
        image1 = simg[sb] # 3 x h x w
        image2 = timg[sb]
        image3 = align_pred12[sb]
        image4 = align_pred21[sb]

        image1 = un_normalize(image1.clone().detach().cpu()).numpy()
        image2 = un_normalize(image2.clone().detach().cpu()).numpy()
        image3 = un_normalize(image3.clone().detach().cpu()).numpy()
        image4 = un_normalize(image4.clone().detach().cpu()).numpy()

        vis_flow12 = flow_to_image(np_flow_12)
        vis_flow21 = flow_to_image(np_flow_21)

        arrows_flow12 = put_optical_flow_arrows_on_image(image1, np_flow_12)
        arrows_flow21 = put_optical_flow_arrows_on_image(image2, np_flow_21)

        occ_mask_12 = (1 - get_occu_mask_bidirection(flow_12.permute(2,0,1).unsqueeze(0), flow_21.permute(2,0,1).unsqueeze(0)).squeeze()).bool().detach()
        occ_mask_21 = (1 - get_occu_mask_bidirection(flow_21.permute(2,0,1).unsqueeze(0), flow_12.permute(2,0,1).unsqueeze(0)).squeeze()).bool().detach()

        masked_image3 = (image3 * (occ_mask_12).unsqueeze(0).detach().cpu().numpy())
        masked_image4 = (image4 * (occ_mask_21).unsqueeze(0).detach().cpu().numpy())

        wandb.log({
            "pair" + str(sb)+"/_ImagePair_1" : [wandb.Image(np.transpose(image1, (1,2,0)))],
            "pair" + str(sb)+"/_ImagePair_2" : [wandb.Image(np.transpose(image2, (1,2,0)))],
            "pair" + str(sb)+"/_ImagePair_proj_12" : [wandb.Image(np.transpose(image3, (1,2,0)))],
            "pair" + str(sb)+"/_ImagePair_proj_21" : [wandb.Image(np.transpose(image4, (1,2,0)))],
            "pair" + str(sb)+"/_ImagePair_proj_12_masked" : [wandb.Image(np.transpose(masked_image3, (1,2,0)))],
            "pair" + str(sb)+"/_ImagePair_proj_21_masked" : [wandb.Image(np.transpose(masked_image4, (1,2,0)))],
            "pair" + str(sb)+"/_flow_image_12" : [wandb.Image(vis_flow12)],
            "pair" + str(sb)+"/_flow_image_21" : [wandb.Image(vis_flow21)],
            "pair" + str(sb)+"/_arrow_flow_12" : [wandb.Image(arrows_flow12)],
            "pair" + str(sb)+"/_arrow_flow_21" : [wandb.Image(arrows_flow21)],
            }, commit=False) 
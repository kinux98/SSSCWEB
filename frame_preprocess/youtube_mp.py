from __future__ import print_function

import os
import numpy as np

import torch
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from model import CRW
from sklearn.ensemble import IsolationForest

from data import youtube
import cv2
import utils
import utils.test_utils as test_utils
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.nn.functional as F
from sklearn.neighbors import LocalOutlierFactor


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([y_base, x_base], 1)  # B2HW
    return base_grid

def imwrite(path, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)

def estimateGaussian(X):
    m = X.shape[0]
    #compute mean of X
    sum_ = np.sum(X,axis=0)
    mu = (sum_/m)
    # compute variance of X
    var = np.var(X,axis=0)
    return mu,var

def multivariateGaussian(X, mu, sigma):
    k = len(mu)
    sigma=np.diag(sigma)
    X = X - mu.T
    p = 1/((2*np.pi)**(k/2)*(np.linalg.det(sigma)**0.5))* np.exp(-0.5* np.sum(X @ np.linalg.pinv(sigma) * X,axis=1))
    return p


def load_model(args):
    model = CRW(args).to(args.device)
    if os.path.isfile(args.resume):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume)
        print(checkpoint['epoch'])
        print(checkpoint['args'])
        if args.model_type == 'scratch':
            state = {}
            for k,v in checkpoint['model'].items():
                if 'conv1.1.weight' in k or 'conv2.1.weight' in k:
                    state[k.replace('.1.weight', '.weight')] = v
                else:
                    state[k] = v
            utils.partial_load(state, model, skip_keys=['head'])
        else:
            utils.partial_load(checkpoint['model'], model, skip_keys=['head'])

        del checkpoint
    
    model.eval()
    model = model.to(args.device)
    return model

def main(args, vis):
    # model = CRW(args, vis=vis).to(args.device)
    # args.mapScale = test_utils.infer_downscale(model)
    model = load_model(args).share_memory()
    args.use_lab = args.model_type == 'uvc'
    dataset = youtube.YoutubeDataset(
        root='/data01/kinux98/SSSCWEB/youtube_download/',
        json_file='/data01/kinux98/SSSCWEB/video_preprocess/video_scene_parsing_new_mt.json',
        image_set='videos_thumbnail_new_test',
        args=args)
   
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    with torch.no_grad():
        test_loss = test(dataset, model, args)

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]      
video_which = '/data01/kinux98/SSSCWEB/youtube_download/videos_thumbnail_new_test'

colors = []
for _ in range(512*512):
    color = np.random.choice(256, 3, replace=True)
    color = (int(color[0]), int(color[1]), int(color[2]))
    colors.append(color)

our_size=256
def mp_test(meta_list, model, args, num):
    n_context=1
    if len(meta_list) > 0:
        base_flow_field = mesh_grid(1, our_size, our_size).long().squeeze().cuda() # 2 * our_size * our_size
        for iidx, meta in tqdm(enumerate(meta_list), total=len(meta_list), leave=False, desc='outer', disable=True):
            print("Num-%d : %d/%d --> %.2f"%(num, iidx+1,len(meta_list), (iidx+1)/len(meta_list)))
            start_idx = meta['start_idx']
            end_idx = meta['end_idx']
            class_name = meta['class_name']
            ytid = meta['ytid']
            shot_idx = meta['shot_idx']

            # print(class_name, ytid, start_idx, end_idx, shot_idx)

            # print('******* Vid %s (%s frames) *******' % (vid_idx, N))
            with torch.inference_mode():
            
                prefix = os.path.join(video_which, class_name, ytid, "frames")
                # print(start_idx, end_idx)
                img_list = []
                for i in range(start_idx, end_idx):
                    img_list.append(i)
                img_list.insert(0, img_list[0])

                lbls = torch.ones(len(img_list) + 1, our_size, our_size).cuda()
                lbls[:n_context] = torch.arange(our_size*our_size).reshape(our_size, our_size).cuda()
                lbls[n_context:] *= -1
                lbls = lbls.long()

                first_img = None
                second_img = None
                for idx, img_index in tqdm(enumerate(img_list), total=len(img_list), desc='inner', leave=False, disable=True):
                    
                    m = img_list[idx:idx+2]
                    # print(m)
                    if len(m) == 1:
                        break
                    fi, si = m
                    src_index = fi
                    target_index = si

                    src_name = "real_{}_save_{}.png".format("{0:010d}".format(src_index), "{0:010d}".format(src_index))
                    target_name = "real_{}_save_{}.png".format("{0:010d}".format(target_index), "{0:010d}".format(target_index))

                    src_path = os.path.join(prefix, src_name)
                    target_path = os.path.join(prefix, target_name)
                    # print(src_path)
                    # print(target_path)
                    src_img = cv2.imread(src_path).astype(np.float32)
                    first_img = cv2.resize(src_img, (256, 256))
                    tgt_img = cv2.imread(target_path).astype(np.float32)
                    second_img = cv2.resize(tgt_img, (256, 256))

                    # (success, saliencyMap) = saliency.computeSaliency(src_img)
                    # saliencyMap = torch.from_numpy(cv2.resize(saliencyMap, (our_size, our_size))).cuda()

                    src_img = cv2.resize(src_img, (512, 512))
                    src_img = src_img / 255.0
                    src_img = src_img[:,:,::-1]
                    src_img = src_img.copy()
                    src_img = youtube.im_to_torch(src_img)
                    src_img = youtube.color_normalize(src_img, mean, std).unsqueeze(0)

                    tgt_img = cv2.resize(tgt_img, (512, 512))
                    tgt_img = tgt_img / 255.0
                    tgt_img = tgt_img[:,:,::-1]
                    tgt_img = tgt_img.copy()
                    tgt_img = youtube.im_to_torch(tgt_img)
                    tgt_img = youtube.color_normalize(tgt_img, mean, std).unsqueeze(0)

                    src_feat = model.encoder(src_img.unsqueeze(1).transpose(1,2).to(args.device))
                    tgt_feat = model.encoder(tgt_img.unsqueeze(1).transpose(1,2).to(args.device))

                    src_feat = F.interpolate(src_feat.squeeze(2), size=(our_size,our_size), mode='bilinear', align_corners=True)
                    tgt_feat = F.interpolate(tgt_feat.squeeze(2), size=(our_size,our_size), mode='bilinear', align_corners=True)

                    # print(src_feat.shape, tgt_feat.shape) # torch.Size([1, 256, 1, our_size, our_size])

                    # print(src_feat.shape, tgt_feat.shape)
                    keys, query = src_feat.flatten(-2), tgt_feat.flatten(-2)

                    Is = test_utils.mem_efficient_batched_affinity_cossim(query, keys, args.topk, args.device)
                    # Is2 = test_utils.batched_affinity_cossim(query, keys, args.topk, args.device)

                    Is = Is[0]

                    ctx_lbls = lbls[idx] 
                    pred_argmax = ctx_lbls.view(-1)[Is.view(-1)].view(our_size,our_size)

                    if idx > 0:
                        base_index = base_flow_field.view(-1, our_size*our_size) # 2 x 4096 [y, x]
                        flatten_Is = Is.view(-1)
                        invalid_mask = (pred_argmax == -99)

                        curr_index = base_index[:,flatten_Is].clone() # 2 x 4096

                        diff0 = torch.abs(base_index[0] - curr_index[0])
                        diff1 = torch.abs(base_index[1] - curr_index[1])

                        diff_npcat = torch.cat([diff0.unsqueeze(1), diff1.unsqueeze(1)], dim=1).cpu().numpy() # 4096 x 2

                        y_pred = IsolationForest(random_state=0).fit_predict(diff_npcat)

                        diff_pred = torch.where(torch.from_numpy(y_pred) == 1, True, False).cuda()

                        change_coord = diff_pred.view(our_size,our_size)
                        change_coord = torch.where(invalid_mask, False, change_coord)

                        if len(change_coord[change_coord==False]) >= 1:
                            pred_argmax[change_coord == False] =  -99 

                        lbls[idx + n_context] = pred_argmax
                        del base_index, flatten_Is, invalid_mask
                        torch.cuda.empty_cache()
                    else:
                        pred_argmax = lbls[0]
                        lbls[idx + n_context] = pred_argmax
                        # curr_index = base_flow_field.view(-1, our_size*our_size)

                    # print("len_of_lbl : ", len(torch.unique(pred_argmax)))
                    if len(torch.unique(pred_argmax)) == 1:
                        break
                    # r_invalid = len(pred_argmax[pred_argmax==-99])/pred_argmax.numel() 
                    # print("ratio of invalid : ", r_invalid)

                    outpath = os.path.join(args.save_path, class_name, ytid, shot_idx)

                    if not os.path.isdir(os.path.join(outpath, "pred")):
                        os.makedirs(os.path.join(outpath, "pred"))

                    test_utils.dump_predictions_yt(
                        pred_argmax, outpath, str(idx))
                    
                    del src_feat, tgt_feat, query, keys, Is, ctx_lbls, pred_argmax

                    torch.cuda.empty_cache()

                    # drawing part
                    # if idx >= (len(img_list)-10):
                    #     direction = 'horizontal'

                    #     flbl_pth = os.path.join(os.path.join(outpath, "pred"), "0_pred.npy")
                    #     first_lbl = np.load(flbl_pth)
                    #     first_lbl_flatten = torch.from_numpy(first_lbl).flatten()

                    #     tlbl_pth = os.path.join(os.path.join(outpath, "pred"), "%d_pred.npy"%(idx))
                    #     target_lbl = torch.from_numpy(np.load(tlbl_pth))
                    #     target_lbl_flatten = target_lbl.flatten()

                    #     intersect = np.intersect1d(first_lbl_flatten, target_lbl_flatten)

                    #     if direction == 'vertical':
                    #         source_image_ = (torch.from_numpy(first_img)).squeeze() # h x w x 3
                    #         target_image_ = (torch.from_numpy(second_img)).squeeze() # h x w x 3
                    #         concat_image = torch.vstack([source_image_,  target_image_]).numpy() # 2h x w x 3
                    #     else:
                    #         source_image_ = (torch.from_numpy(first_img)).squeeze() # h x w x 3
                    #         target_image_ = (torch.from_numpy(second_img)).squeeze() # h x w x 3
                    #         concat_image = torch.hstack([source_image_,  target_image_]).numpy() # h x 2w x 3
                        
                    #     concat_untaced = concat_image.copy()

                    #     candid = torch.nonzero((target_lbl != -99), as_tuple=False)
                    #     np.random.seed(20170890)
                    #     # elected_candidate = np.random.choice(intersect, min(50, len(candid)-1), replace=True)
                    #     elected_candidate = np.random.choice(intersect, len(intersect)-1, replace=False)

                    #     for ccnt, rnd_sampled_idx in enumerate(elected_candidate):

                    #         k = torch.nonzero(torch.from_numpy(first_lbl) == rnd_sampled_idx,as_tuple=False)
                    #         base_y, base_x = k[0]
                    #         base_y = base_y.item()
                    #         base_x = base_x.item()

                    #         base_y = int(base_y * 256/our_size)
                    #         base_x = int(base_x * 256/our_size)

                    #         corr_candid = torch.nonzero(target_lbl == rnd_sampled_idx, as_tuple=False)
                    #         # elected_candidate_corr = np.random.choice(len(corr_candid), min(10, len(corr_candid)-1), replace=False)
                    #         elected_candidate_corr = np.random.choice(len(corr_candid), len(corr_candid), replace=False)

                    #         for cccnt, rnd_sampled_idx_corr in enumerate(elected_candidate_corr):

                    #             corr_coord = corr_candid[rnd_sampled_idx_corr]
                    #             wrpd_y = corr_coord[0].item()
                    #             wrpd_x = corr_coord[1].item()

                    #             wrpd_y = int(wrpd_y * 256/our_size)
                    #             wrpd_x = int(wrpd_x * 256/our_size)

                    #             dest_coord = (base_x, base_y)
                    #             source_coord = (wrpd_x+256, wrpd_y)

                    #             # print(source_coord, dest_coord)
                    #             cv2.line(concat_image, source_coord, dest_coord, colors[rnd_sampled_idx_corr], thickness=1, lineType=cv2.LINE_AA)

                    #     path_ = os.path.join(outpath , "draw_line")
                    #     if not os.path.isdir(path_):
                    #         os.makedirs(path_)
                    #     final_path = os.path.join(path_, "img_"+str(direction)+str(0)+"_to_"+str(idx)+".jpg")
                    #     print(final_path)
                    #     imwrite(final_path, concat_image)
                    #     imwrite(os.path.join(outpath , "origin", "img_"+str(direction)+str(0)+"_to_"+str(idx)+".jpg"), concat_untaced)
                        
                del img_list, lbls
                torch.cuda.empty_cache()



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def test(loader, model, args):
    candid_data = []
    for vid_idx, meta in enumerate(loader):
        outpath = os.path.join(args.save_path, meta['class_name'], meta['ytid'], meta['shot_idx'])

        if not os.path.isdir(outpath):
            # print(meta['class_name'], meta['ytid'], meta['shot_idx'])
            candid_data.append(meta)

    print("TOTAL : ", len(candid_data))
    candid_data = list(chunks(candid_data, len(candid_data))) # [ 0, 1, 2, 3, 4, ...]
    n_proc = 30
    for c, scd in tqdm(enumerate(candid_data), desc='total', total=len(candid_data), leave=False):
        print("LEVEL : ", c+1, len(candid_data))
        # scd : [0, 1, 2, 3, ... , 4999]
        mp_candid_data_list = list(split(scd, n_proc))
        for idx, j in enumerate(mp_candid_data_list):
            print(idx, len(mp_candid_data_list[idx]), len(j))

        procs = []
        for i in range(n_proc):
            p = mp.Process(
                target=mp_test, 
                args=(mp_candid_data_list[i], model, args, i)
            )
            procs.append(p)
            p.start()

        for idx, sp in enumerate(procs):
            sp.join()
        
        torch.cuda.empty_cache()
        
# python youtube_mp.py --resume=./checkpoints/youtube_consecutive/checkpoint.pth     
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = utils.arguments.test_args()
    args.imgSize = args.cropSize
    print('Context Length:', args.videoLen, 'Image Size:', args.imgSize)
    print('Arguments', args)

    main(args, None)

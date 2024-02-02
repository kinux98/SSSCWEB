r'''
    modified training script of GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''

import argparse
import copy
from lib2to3.pytree import convert
import os
import pickle
import random
import time
from os import path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from data.youtube import YoutubeDataset
from termcolor import colored
from torch.utils.data import DataLoader

from models.cats import CATs, Discriminator
import utils_training.optimize as optimize
from utils_training.evaluation import Evaluator
from utils_training.utils import parse_list, load_checkpoint, save_checkpoint, boolean_string
from data import download
import gc
from torch.cuda.amp import autocast
import wandb
import warnings
warnings.filterwarnings(action='ignore')

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    # torch.autograd.set_detect_anomaly(True)
    # Argument parsing
    parser = argparse.ArgumentParser(description='CATs Training Script')
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--pretrained', dest='pretrained', default=None,
                       help='path to pre-trained model')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='start epoch')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=48,
                        help='training batch size')
    parser.add_argument('--batch-size-yt', type=int, default=24,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=32,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Pseudo-RNG seed')
                        
    parser.add_argument('--datapath', type=str, default='./Datasets_CATs')
    parser.add_argument('--benchmark', type=str, default='spair', choices=['pfpascal', 'spair'])
    parser.add_argument('--eval_benchmark', type=str, default='spair', choices=['pfpascal', 'spair', 'pfwillow'])
    parser.add_argument('--eval_benchmark2', type=str, default=None, choices=['pfpascal', 'spair', 'pfwillow'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',
                        help='learning rate (default: 3e-5)')
    parser.add_argument('--lr-backbone', type=float, default=3e-6, metavar='LR',
                        help='learning rate (default: 3e-6)')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine'])
    parser.add_argument('--step', type=str, default='[1700, 1800, 1900]') # 
    parser.add_argument('--step_gamma', type=float, default=0.5)

    parser.add_argument('--feature-size', type=int, default=16)
    parser.add_argument('--feature-proj-dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--mlp-ratio', type=int, default=4)
    parser.add_argument('--hyperpixel', type=str, default='[0,8,20,21,26,28,29,30]')
    parser.add_argument('--freeze', type=boolean_string, nargs='?', const=True, default=False)
    parser.add_argument('--augmentation', type=boolean_string, nargs='?', const=True, default=True)
    
    parser.add_argument('--run_yt', type=boolean_string, nargs='?', const=True, default=True) # running yt dataset
    parser.add_argument('--run_sb', type=boolean_string, nargs='?', const=True, default=True) # running standard benchmark
    parser.add_argument('--run_dann', type=boolean_string, nargs='?', const=True, default=True) # running standard benchmark
    parser.add_argument('--run_contra', type=boolean_string, nargs='?', const=True, default=False)
    args = parser.parse_args()
    
    if args.run_yt == True:
        print("running YT")
    if args.run_sb == True:
        print("running SB")  
    # Seed
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Evaluator
    Evaluator.initialize(args.benchmark, args.alpha)
    
    # Dataloader
    download.download_dataset(args.datapath, args.benchmark)

    train_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'trn', args.augmentation, args.feature_size)
    
    val_dataset = download.load_dataset(args.eval_benchmark, args.datapath, args.thres, device, 'test', args.augmentation, args.feature_size)
    
    train_dataloader = DataLoader(train_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_threads,
        shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_threads,
        shuffle=False)
    
    if args.eval_benchmark2 != None:
        val_dataset2 = download.load_dataset(args.eval_benchmark2, args.datapath, args.thres, device, 'test', args.augmentation, args.feature_size)
    
        val_dataloader2 = DataLoader(val_dataset2,
            batch_size=args.batch_size,
            num_workers=args.n_threads,
            shuffle=False)
    
    youtube_dataset = YoutubeDataset(
        image_set="/your/path/SSSCWEB/youtube_download/videos_thumbnail_new_test",
        json_file="/your/path/SSSCWEB/training/videowalk/code/Cost-Aggregation-transformers/video_scene_parsing_new_mt.json",
        pl_path="/your/path/SSSCWEB/SSSCWEB/frame_preprocess/results/",
        feature_size=args.feature_size
    )

    youtube_dataloader = DataLoader(youtube_dataset,
        batch_size=args.batch_size_yt,
        num_workers=args.n_threads,
        shuffle=True, drop_last=True)

    print("Youtube dataset length : ", len(youtube_dataloader))
    youtube_dataloader = iter(cycle(youtube_dataloader))

    
    # Model
    if args.freeze:
        print('Backbone frozen!')
    model = CATs(
        feature_size=args.feature_size, feature_proj_dim=args.feature_proj_dim,
        depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
        hyperpixel_ids=parse_list(args.hyperpixel), freeze=False)

    param_model = [param for name, param in model.named_parameters() if 'feature_extraction' not in name]
    param_backbone = [param for name, param in model.named_parameters() if 'feature_extraction' in name]

    # if args.run_dann == True:
    da_disc = Discriminator()
    
    # Optimizer
    params = [
        {'params': param_model, 'lr': args.lr}, 
        {'params': param_backbone, 'lr': args.lr_backbone},
    ]

    if args.run_dann == True:
        params.append({'params': da_disc.parameters(), 'lr': args.lr})
    
    optimizer = optim.AdamW(params, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() 
    # Scheduler
    scheduler = None
    scheduler = \
        lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6, verbose=True)\
        if args.scheduler == 'cosine' else\
        lr_scheduler.MultiStepLR(optimizer, milestones=parse_list(args.step), gamma=args.step_gamma, verbose=True)

    if args.pretrained:
        # reload from pre_trained_model
        model, da_disc, scaler, optimizer, scheduler, start_epoch, best_val = load_checkpoint(model, da_disc, scaler,optimizer, scheduler,
                                                                 filename=args.pretrained)
        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        cur_snapshot = os.path.basename(os.path.dirname(args.pretrained))

    else:
        best_val = 0
        start_epoch = 0

    # start_epoch = 0

    if not os.path.isdir(args.snapshots):
        os.makedirs(args.snapshots)
    cur_snapshot = args.name_exp
    if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
        os.makedirs(osp.join(args.snapshots, cur_snapshot))
    with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    
    # create summary writer
    save_path=osp.join(args.snapshots, cur_snapshot)
    print("save path : ", save_path)
    
    t_model = None
    model = nn.DataParallel(model)
    model = model.to(device)

    if args.run_dann == True:
        da_disc = nn.DataParallel(da_disc)
        da_disc = da_disc.to(device)
    else:
        da_disc = None
    
    train_started = time.time()
    is_sgood = True

    wandb.init(project="SSSCWEB")#, mode="disabled")
    wandb.run.name = str(args.snapshots.split("/")[-1])

    for epoch in range(start_epoch, args.epochs):
        with autocast():
            train_loss = optimize.train_epoch(
                                         model, 
                                         t_model, is_sgood, scaler, da_disc,
                                         optimizer,
                                         train_dataloader,
                                         youtube_dataloader,
                                         device,
                                         epoch, args)
                                        #  train_writer)
        wandb.log({
            "loss/train_loss_per_epoch" : train_loss,
            "optim/LR" :  scheduler.get_lr()[0], 
            "optim/LR_backbone" : scheduler.get_lr()[1],
            "epoch" : epoch
        })

        val_loss_grid, val_mean_pck = optimize.validate_epoch(model,
                                                       val_dataloader,
                                                       device,
                                                       epoch=epoch)
        wandb.log({
            "val/mean PCK student" : val_mean_pck,
            "val/loss grid student" :  val_loss_grid, 
        })
        is_best = val_mean_pck > best_val
        best_val = max(val_mean_pck, best_val)

        if t_model != None:
            t_val_loss_grid, t_val_mean_pck = optimize.validate_epoch(t_model,
                                                           val_dataloader,
                                                           device,
                                                           epoch=epoch)
            wandb.log({
                "val/mean PCK teacher" :t_val_mean_pck,
                "val/loss grid teacher" :  t_val_loss_grid, 
            })
        is_best = val_mean_pck > best_val
        best_val = max(val_mean_pck, best_val)

        if args.eval_benchmark2 != None:
            val_loss_grid, val_mean_pck = optimize.validate_epoch(model,
                                                       val_dataloader2,
                                                       device,
                                                       epoch=epoch)
            wandb.log({
                "val/mean PCK student v2" : val_mean_pck,
                "val/loss grid student v2" :  val_loss_grid, 
            })

        if args.eval_benchmark == 'spair':
            if epoch % 1 == 0:
                d = {'epoch': epoch + 1,
                                 'state_dict': model.module.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'scaler' : scaler.state_dict(),
                                 'scheduler': scheduler.state_dict(),
                                 'best_loss': best_val}
                if da_disc != None:
                    d['da_disc'] = da_disc.module.state_dict()

                save_checkpoint(d, is_best, save_path, 'epoch_{}.pth'.format(epoch + 1))
        else:
            if epoch % 10 == 0:
                d = {'epoch': epoch + 1,
                                 'state_dict': model.module.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'scaler' : scaler.state_dict(),
                                 'scheduler': scheduler.state_dict(),
                                 'best_loss': best_val}
                if da_disc != None:
                    d['da_disc'] = da_disc.module.state_dict()

                save_checkpoint(d, is_best, save_path, 'epoch_{}.pth'.format(epoch + 1))
        


        scheduler.step(epoch)
        del train_loss, val_loss_grid, val_mean_pck
        torch.cuda.empty_cache()
        gc.collect()

    print(args.seed, 'Training took:', time.time()-train_started, 'seconds')

r'''
    modified test script of GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''

import argparse
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
from tensorboardX import SummaryWriter
from termcolor import colored
from torch.utils.data import DataLoader

from models.cats import CATs
from models.cats_gelu import CATs_gelu
import utils_training.optimize as optimize
from utils_training.evaluation import Evaluator
from utils_training.utils import parse_list, log_args, load_checkpoint, save_checkpoint, boolean_string
from data import download


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='CATs Test Script')
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--snapshots', type=str, default='./eval')
    parser.add_argument('--pretrained', dest='pretrained',
                       help='path to pre-trained model')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=32,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Pseudo-RNG seed')
                        
    parser.add_argument('--datapath', type=str, default='./Datasets_CATs')
    parser.add_argument('--benchmark', type=str, choices=['pfpascal', 'spair', 'pfwillow'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--feature-size', type=int, default=16)
    parser.add_argument('--feature-proj-dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--num-heads', type=int, default=6)
    parser.add_argument('--mlp-ratio', type=int, default=4)
    parser.add_argument('--hyperpixel', type=str, default='[0,8,20,21,26,28,29,30]')
    

    # Seed
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize Evaluator
    Evaluator.initialize(args.benchmark, args.alpha)
    
    # with open(osp.join(args.pretrained, 'args.pkl'), 'rb') as f:
    #     args_model = pickle.load(f)
    # log_args(args_model)
    
    # Dataloader
    download.download_dataset(args.datapath, args.benchmark)
    test_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'test', False, args.feature_size)
    test_dataloader = DataLoader(test_dataset,
        batch_size=128,
        num_workers=args.n_threads,
        shuffle=False)

    # Model
    def load_state_dict(model, pretrain):
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in pretrain.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
    model = CATs_gelu(
        feature_size=args.feature_size, feature_proj_dim=args.feature_proj_dim,
        depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
        hyperpixel_ids=parse_list(args.hyperpixel), freeze=True)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        # model.load_state_dict(checkpoint['state_dict'])
        load_state_dict(model, checkpoint['state_dict'])
    else:
        raise NotImplementedError()
    # create summary writer

    model = nn.DataParallel(model)
    model = model.to(device).eval()

    train_started = time.time()

    val_loss_grid, val_mean_pck = optimize.validate_epoch(model,
                                                    test_dataloader,
                                                    device,
                                                    epoch=0)
    # tot_n = 0
    # tot_pck = 0
    # for k in class_dict.keys():
    #     tot_n += len(class_dict[k])
    #     tot_pck += np.sum(class_dict[k])
    #     print(k, np.mean(class_dict[k]))

    # print("avg PCK : ", tot_pck / tot_n)

    print(colored('==> ', 'blue') + 'Test average grid loss :',
            val_loss_grid)
    print('mean PCK is {}'.format(val_mean_pck))

    print(args.seed, 'Test took:', time.time()-train_started, 'seconds')

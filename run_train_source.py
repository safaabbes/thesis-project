import os
import argparse
import sys
import pathlib
import shutil

import torchvision
from torchvision import transforms 
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import numpy as np

from utils import *
from datasets import *
from train_epochs import *
from resnet import *

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)

def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment identifer
    parser.add_argument('--exp', type=str, required=True, help='Experiment Number')
    parser.add_argument('--device', type=str, default='cuda', help='Computational Device')
    # Source and target domain
    parser.add_argument('--path', type=str, default='/storage/TEV/sabbes/domainnet40/', help='Dataset Path')
    parser.add_argument('--source', type=str, required=True, help='Source Domain Name')
    parser.add_argument('--target', type=str, required=True, help='Target Domain Name')
    # Training details	
    parser.add_argument('--bs', type=int, default= 16, help='Batch Size')
    parser.add_argument('--n_epochs', type=int, default= 25, help='Number of Epochs')
    parser.add_argument('--lr', type=float, default= 1e-3, help='Learning Rate')
    parser.add_argument('--optimizer', type=str, default= 'SGD', help='Optimizer')
    parser.add_argument('--scheduler', type=str, default= 'cosine', help='Scheduler')
    parser.add_argument('--step', type=int, default= 20, help='Step for the learning rate decay')
    parser.add_argument('--momentum', type=float, default= 0.9, help='Momentum')
    parser.add_argument('--w_decay', type=float, default= 1e-4, help='Weight Decay')
    
    args=parser.parse_args()
    return args

def main():
    
    args = parse_args()    
    path_log= os.path.join('../logs', 'train_{}.log'.format(args.exp))
    logger = setup_logger(path_log)

    # for arg, value in vars(args).items():
    # logger.info('{} = {}'.format(arg, value))
        
    ################################################################################################################
    #### Setup source data loaders
    ################################################################################################################
    
    source_ds = DomainNetDataset40(args.source, args.path)
    s_train_ds, s_test_ds = source_ds.get_dataset()
    s_train_dl, s_test_dl = source_ds.get_dataloaders(train_ds = s_train_ds, test_ds = s_test_ds, batch_size = args.bs)

    logger.info('number of train samples of {} dataset: {}'.format(args.source, len(s_train_ds)))
    logger.info('number of test samples of {} dataset: {}'.format(args.source, len(s_test_ds)))
    
    logger.info('len of {} train dataloader {}'.format(args.source, len(s_train_dl)))
    logger.info('len of {} test dataloader {}'.format(args.source, len(s_test_dl)))
    
    s_train_dl = DeviceDataLoader(s_train_dl, args.device)
    s_test_dl = DeviceDataLoader(s_test_dl, args.device)

    ################################################################################################################
    #### Setup target data loaders
    ################################################################################################################
    target_ds = DomainNetDataset40(args.target, args.path)
    
    t_train_ds, t_test_ds = target_ds.get_dataset()
    t_train_dl, t_test_dl = target_ds.get_dataloaders(train_ds = t_train_ds, test_ds = t_test_ds, batch_size = args.bs)

    logger.info('number of train samples of {} dataset: {}'.format(args.target, len(t_train_ds)))
    logger.info('number of test samples of {} dataset: {}'.format(args.target, len(t_test_ds)))
    
    logger.info('len of {} train dataloader {}'.format(args.target, len(t_train_dl)))
    logger.info('len of {} test dataloader {}'.format(args.target, len(t_test_dl)))
    
    t_train_dl = DeviceDataLoader(t_train_dl, args.device)
    t_test_dl = DeviceDataLoader(t_test_dl, args.device)
    
    ################################################################################################################
    #### Setup model	 
    ################################################################################################################
    
    # Testing with SENTRY's Resnet50
    # model = SENTRY_ResNet50() 
    # model = model.to(args.device, non_blocking= True)
    # logger.info('Using SENTRY ResNet50')
    
    # Testing with pre-trained Pytorch Resnet50 with fc reinitialized 
    # model = ResNet50(num_classes=40, pre_trained=True)
    # model = model.to(args.device, non_blocking= True)
    # logger.info('Using Normal ResNet50')
    
    # Testing New architecture
    model = SC_Res50(num_classes=40, n_super_classes=5)
    model = model.to(args.device, non_blocking= True)
    logger.info('Using New Architecture')
    
    
    
    ################################################################################################################
    #### Setup Training	 
    ################################################################################################################
    
    # setup optimizer
    # TASK: Since we're using pre-trained models, there will need to be a separation between pretraied and re-initialized layers
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            params = model.parameters(),
            lr=args.lr,
            momentum = args.momentum,
            weight_decay = args.w_decay
        )
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(
            params = model.parameters(),
            lr=args.lr,
            weight_decay = args.w_decay,
        )
    # ADD OTHER Elif: AdamW
    else:
        logger.warning('No optimizer specified!')
        optimizer = None
        
    # setup scheduler
    if args.step is None:
        args.step = args.n_epochs
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer = optimizer,
            T_max = args.step, #Maximum number of iterations
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer= optimizer,
            step_size = args.step,
        )
    else:
        logger.warning('No scheduler specified!')
        scheduler = None   
        
    # Test and Train over epochs
    run_epochs(
               s_train_dl, s_test_dl, 
               t_train_dl, t_test_dl,
               model,
               args,
               optimizer,
               scheduler,
               logger
             )


if __name__ == '__main__':
    main()
    
    
    
    
    # REDUCED DATASET FOR DEBUGGING
    # s_remove_size = int(0.98 * len(s_train_ds))
    # t_remove_size = int(0.98 * len(s_test_ds))
    # s_train_ds, _ = random_split(s_train_ds, [len(s_train_ds) - s_remove_size, s_remove_size])  
    # s_test_ds, _ = random_split(s_test_ds, [len(s_test_ds) - t_remove_size, t_remove_size]) 
    # s_train_dl = torch.utils.data.DataLoader(s_train_ds, batch_size=args.bs, shuffle=True)
    # s_test_dl = torch.utils.data.DataLoader(s_test_ds, batch_size=args.bs*2)          
    
    # REDUCED DATASET FOR DEBUGGING
    # s_remove_size = int(0.98 * len(t_train_ds))
    # t_remove_size = int(0.98 * len(t_test_ds))
    # t_train_ds, _ = random_split(t_train_ds, [len(t_train_ds) - s_remove_size, s_remove_size])  
    # t_test_ds, _ = random_split(t_test_ds, [len(t_test_ds) - t_remove_size, t_remove_size]) 
    # t_train_dl = torch.utils.data.DataLoader(t_train_ds, batch_size=args.bs, shuffle=True)
    # t_test_dl = torch.utils.data.DataLoader(t_test_ds, batch_size=args.bs*2)        
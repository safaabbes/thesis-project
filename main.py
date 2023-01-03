import os
import argparse
import sys
import pathlib
import shutil
import distutils

from torchsummary import summary
import torchvision
from torchvision import transforms 
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import numpy as np
import torch.nn as nn

from utils import *
from datasets import *
from train_baseline import *
from train_model import *
from resnet import *

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp', type=str, required=True, help='Experiment Number')
    parser.add_argument('--device', type=str, default='cuda', help='Computational Device')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234) # Similar to SENTRY's Seed
    parser.add_argument('--freq_saving', type=int, default=5)
    
    # Model
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.05)
    
    # Train
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=40)
    parser.add_argument('--balance_mini_batches', type=lambda x:bool(distutils.util.strtobool(x)), required=True)
    
    # Data
    parser.add_argument('--path', type=str, default='/storage/TEV/sabbes/domainnet40/', help='Dataset Path')
    parser.add_argument('--source', type=str, required=True, help='Source Domain Name')
    parser.add_argument('--target', type=str, required=True, help='Target Domain Name')
    
    # Loss
    parser.add_argument('--reduction', type=str, default='sum')
    parser.add_argument('--mu1', type=float, default= 0.33, help='Weight of the loss of Main Branch')	
    parser.add_argument('--mu2', type=float, default= 0.33, help='Weight of the loss of Source Branch')
    parser.add_argument('--mu3', type=float, default= 0.33, help='Weight of the loss of Target Branch')
    
    # Fixed Hyperparameters for Consistenty with the current baseline and previous papers (SENTRY)
    parser.add_argument('--lr', type=float, default= 1e-3, help='Learning Rate')
    parser.add_argument('--optimizer', type=str, default= 'SGD', help='Optimizer')
    parser.add_argument('--momentum', type=float, default= 0.9, help='Momentum')
    parser.add_argument('--wd', type=float, default= 5e-4, help='Weight Decay')
    
    args=parser.parse_args()
    return args

def main():
    
    # Parse input arguments
    args = parse_args()
    
    # Create logger
    path_log= os.path.join('../logs', 'train_{}.log'.format(args.exp))
    logger = setup_logger(path_log)

    # Log library versions
    logger.info('PyTorch version = {:s}'.format(torch.__version__))
    logger.info('TorchVision version = {:s}'.format(torchvision.__version__))
    # Activate CUDNN backend (https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936)
    torch.backends.cudnn.enabled = True

    # Fix random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        
    # Log input arguments
    # for arg, value in vars(args).items():
    #     logger.info('{:s} = {:s}'.format(arg, str(value)))
                
    ################################################################################################################
    #### Setup Source DataLoaders
    ################################################################################################################
    
    source_ds = DomainNetDataset40(args.source, args.path)
    s_train_ds, s_test_ds = source_ds.get_dataset()
    
    if args.balance_mini_batches == True:
        s_train_idx = np.arange(len(s_train_ds))
        y_train = [s_train_ds.targets[i] for i in s_train_idx]
        count = dict(Counter(s_train_ds.targets))
        class_sample_count = np.array(list(count.values()))
        # Find weights for each class
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for (t,_,_) in y_train])
        samples_weight = torch.from_numpy(samples_weight)
        # Create Weighted Random Sampler
        train_sampler = WeightedRandomSampler(
        weights= samples_weight.type('torch.DoubleTensor'),
        num_samples= len(samples_weight),)
        s_train_dl = torch.utils.data.DataLoader(
            dataset=s_train_ds, 
            batch_size=args.bs, 
            sampler=train_sampler, 
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True)
    else:
        s_train_dl = torch.utils.data.DataLoader(
            dataset=s_train_ds, 
            batch_size=args.bs, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True)
        
    s_test_dl = torch.utils.data.DataLoader(
        dataset=s_test_ds, 
        batch_size=args.bs, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)

    logger.info('number of train samples of {} dataset: {}'.format(args.source, len(s_train_ds)))
    logger.info('number of test samples of {} dataset: {}'.format(args.source, len(s_test_ds)))
    logger.info('len of {} train dataloader {}'.format(args.source, len(s_train_dl)))
    logger.info('len of {} test dataloader {}'.format(args.source, len(s_test_dl)))
    
    # Move Data to GPU
    s_train_dl = DeviceDataLoader(s_train_dl, args.device)
    s_test_dl = DeviceDataLoader(s_test_dl, args.device)

    ################################################################################################################
    #### Setup Target DataLoaders
    ################################################################################################################
    
    target_ds = DomainNetDataset40(args.target, args.path)
    
    t_train_ds, t_test_ds = target_ds.get_dataset()
    
    if args.balance_mini_batches == True:
        t_train_idx = np.arange(len(t_train_ds))
        y_train = [t_train_ds.targets[i] for i in t_train_idx]
        count = dict(Counter(t_train_ds.targets))
        class_sample_count = np.array(list(count.values()))
        # Find weights for each class
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for (t,_,_) in y_train])
        samples_weight = torch.from_numpy(samples_weight)
        # Create Weighted Random Sampler
        train_sampler = WeightedRandomSampler(
        weights= samples_weight.type('torch.DoubleTensor'),
        num_samples= len(samples_weight),)
        t_train_dl = torch.utils.data.DataLoader(
            dataset=t_train_ds, 
            batch_size=args.bs, 
            sampler=train_sampler, 
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True)
    else:
        t_train_dl = torch.utils.data.DataLoader(
            dataset=t_train_ds, 
            batch_size=args.bs, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True)
        
    t_test_dl = torch.utils.data.DataLoader(
        dataset=t_test_ds, 
        batch_size=args.bs, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)
    
    logger.info('number of train samples of {} dataset: {}'.format(args.target, len(t_train_ds)))
    logger.info('number of test samples of {} dataset: {}'.format(args.target, len(t_test_ds)))
    logger.info('len of {} train dataloader {}'.format(args.target, len(t_train_dl)))
    logger.info('len of {} test dataloader {}'.format(args.target, len(t_test_dl)))
    
    # Move Data to GPU
    t_train_dl = DeviceDataLoader(t_train_dl, args.device)
    t_test_dl = DeviceDataLoader(t_test_dl, args.device)
    
    ################################################################################################################
    #### Setup Model	 
    ################################################################################################################
    
    if args.model_type == 'R50_2H':
        model = resnet50s(args) 
        model = model.to(args.device)
        logger.info('Using Model with 2 heads')
    elif args.model_type == 'R50_1H':
        model = resnet50s_1head(args) 
        model = model.to(args.device)
        logger.info('Using Model with 1 head')
    else:
        raise ValueError("Model Type Not Specified!")
    
    # Set data parallelism
    if torch.cuda.device_count() == 1:
        logger.info('Using a single GPU, data parallelism is disabled')
    else:
        logger.info('Using multiple GPUs, with data parallelism')
        model = torch.nn.DataParallel(model)

    # Get the model summary
    # if torch.cuda.device_count() == 1:
    #     logger.info('Model summary:')
    #     stats = torchinfo.summary(model, (args.bs, 3, 128, 128))
    #     logger.info(str(stats))
    
    ################################################################################################################
    #### Run Train 
    ################################################################################################################
    
    # Setup optimizer
    if args.model_type == 'R50_2H':
        head = ['head1.weight', 'head1.bias', 'head2.weight', 'head2.bias']
    elif args.model_type == 'R50_1H':
        head = ['head.weight', 'head.bias']
        
    params_head = list(filter(lambda kv: kv[0] in head, model.named_parameters()))
    params_back = list(filter(lambda kv: kv[0] not in head, model.named_parameters()))
    
    # logger.info('Learnable backbone parameters:')
    # for name, param in params_back:
    #     if param.requires_grad is True:
    #         logger.info(name)
    # logger.info('Learnable head parameters:')
    # for name, param in params_head:
    #     if param.requires_grad is True:
    #         logger.info(name)
            
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            [
                {'params': [p for n, p in params_back], 'lr': 0.1 * args.lr},
                {'params': [p for n, p in params_head], 'lr': args.lr}
            ],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
            nesterov=True)
    else:
        raise NotImplementedError
    
    
    # Get losses and send them to the device
    criterion = nn.CrossEntropyLoss(reduction=args.reduction)
    criterion = criterion.to(args.device)
    
    train_model(
                s_train_dl, s_test_dl, 
                t_train_dl, t_test_dl,
                model, args,
                optimizer, criterion,
                logger
                )

if __name__ == '__main__':
    main()

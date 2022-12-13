import os
import argparse
import sys
import pathlib
import shutil

from torchsummary import summary
import torchvision
from torchvision import transforms 
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import numpy as np

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
    parser.add_argument('--task', type=str, default='run_model_v1',help='Run Baseline or New Model')
    parser.add_argument('--freq_saving', type=int, default=10)

    # Train
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--cbt', type=str, default= 'True', help='Class Balance Train')
    
    # Data
    parser.add_argument('--path', type=str, default='/storage/TEV/sabbes/domainnet40/', help='Dataset Path')
    parser.add_argument('--source', type=str, required=True, help='Source Domain Name')
    parser.add_argument('--target', type=str, required=True, help='Target Domain Name')
    
    # Super Classes Hyper-parameters	
    parser.add_argument('--alpha', type=float, default= 0.5, help='Weight of the loss of Source Branch')
    parser.add_argument('--gamma', type=float, default= 0.5, help='Weight of the loss of Target Branch')
    
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
    s_train_dl, s_test_dl = source_ds.get_dataloaders(train_ds = s_train_ds, test_ds = s_test_ds, batch_size = args.bs , class_balance_train = args.cbt)

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
    t_train_dl, t_test_dl = target_ds.get_dataloaders(train_ds = t_train_ds, test_ds = t_test_ds, batch_size = args.bs)

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
    
    if args.task == 'run_sentry_baseline':
        # Using SENTRY's Resnet50
        model = SENTRY_ResNet50() 
        model = model.to(args.device, non_blocking= True)
        logger.info('Baseline Training with SENTRY ResNet50')
    elif args.task == 'run_original_baseline':
        # Using Original Resnet50
        model = ResNet50(num_classes=40) 
        model = model.to(args.device, non_blocking= True)
        logger.info('Baseline Training with Original ResNet50')
    elif args.task == 'run_model_v1':
        # Testing Model v1
        model = Res50_V1(num_classes=40, n_super_classes=5)
        model = model.to(args.device, non_blocking= True)
        logger.info('Using Super-Classes Model V1')
    elif args.task == 'run_model_v2':
        # Testing Model v2
        model = Res50_V1(num_classes=40, n_super_classes=13)
        model = model.to(args.device, non_blocking= True)
        logger.info('Using Super-Classes Model V2')
    else:
        raise ValueError("Task Not Specified!")
    
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
    
    # Setup SENTRY's Optimizer
    optimizer = generate_optimizer(model, args)
    
    # Test and Train over epochs
    if args.task == 'run_sentry_baseline' or args.task == 'run_original_baseline':
        train_baseline(
                s_train_dl, s_test_dl, 
                t_train_dl, t_test_dl,
                model,
                args,
                optimizer,
                logger
                )
    else:
        train_model(
                s_train_dl, s_test_dl, 
                t_train_dl, t_test_dl,
                model,
                args,
                optimizer,
                logger
                )

if __name__ == '__main__':
    main()

import os
import argparse
import sys
import pathlib
import shutil

import torchvision
from torchvision import transforms 
from torchvision.datasets import ImageFolder
# from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim
import matplotlib.pyplot as plt

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
    parser.add_argument('--bs', type=int, default= 64, help='Batch Size')
    parser.add_argument('--n_epochs', type=int, default= 10, help='Number of Epochs')
    parser.add_argument('--lr', type=float, default= 1e-3, help='Learning Rate')
    parser.add_argument('--optimizer', type=str, default= 'SGD', help='Optimizer')
    parser.add_argument('--scheduler', type=str, default= 'cosine', help='Scheduler')
    parser.add_argument('--step', type=int, default= None, help='Step for the learning rate decay')
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
    
    s_train_dl, s_test_dl = source_ds.get_dataloaders(batch_size = args.bs)
    
    logger.info('number of train samples of {} dataset: {}'.format(args.source, source_ds.train_size))
    logger.info('number of test samples of {} dataset: {}'.format(args.source, source_ds.test_size))
    
    logger.info('len of {} train dataloader {}'.format(args.source, len(s_train_dl)))
    logger.info('len of {} test dataloader {}'.format(args.source, len(s_test_dl)))
    
    s_train_dl = DeviceDataLoader(s_train_dl, args.device)
    s_test_dl = DeviceDataLoader(s_test_dl, args.device)

    ################################################################################################################
    #### Setup target data loaders
    ################################################################################################################
    target_ds = DomainNetDataset40(args.target, args.path)
    
    t_train_dl, t_test_dl = target_ds.get_dataloaders(batch_size = args.bs)
    
    logger.info('number of train samples of {} dataset: {}'.format(args.target, target_ds.train_size))
    logger.info('number of test samples of {} dataset: {}'.format(args.target, target_ds.test_size))
    
    logger.info('len of {} train dataloader {}'.format(args.target, len(t_train_dl)))
    logger.info('len of {} test dataloader {}'.format(args.target, len(t_test_dl)))
    
    t_train_dl = DeviceDataLoader(t_train_dl, args.device)
    t_test_dl = DeviceDataLoader(t_test_dl, args.device)
    
    ################################################################################################################
    #### Setup model	 
    ################################################################################################################
       
    model = ResNet50() 
    model = model.to(args.device, non_blocking= True)
    
    #load model and send it to device    
    # weights = ResNet50_Weights.DEFAULT   
    # model = resnet50(weights=weights)  
    # model = model.to(args.device, non_blocking= True)
    
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
    run_epochs(s_train_dl, s_test_dl, 
               t_train_dl, t_test_dl,
               model,
               args,
               optimizer,
               scheduler,
               logger
             )


if __name__ == '__main__':
    main()
    
    
    
    
              
    # source_transforms = [transforms.Resize((224, 224)),
    #                     #  transforms.CenterCrop(224),
    #                      transforms.ToTensor(),
    #                     #  transforms.Normalize(*dataset_stats[args.source_name]),
    #                      ]
    # target_transforms = [transforms.Resize((224, 224)),
    #                     # transforms.CenterCrop(224),
    #                     transforms.ToTensor(),
    #                     # transforms.Normalize(*dataset_stats[args.target_name]),
    #                     ]
    
    # img_transform_source = transforms.Compose(source_transforms)
    # img_transform_target = transforms.Compose(target_transforms)
    
    # source_ds = ImageFolder(args.path+args.source_name, img_transform_source)
    # target_ds = ImageFolder(args.path+args.target_name, img_transform_target)
    
    # logger.info('number of {} samples: {}'.format(args.source_name , len(source_ds)))
    # logger.info('number of {} samples: {}'.format(args.target_name , len(target_ds)))
    # logger.info('Does the source and target have the same labels?: {}'.format(source_ds.classes == target_ds.classes))
    
    # ONGOING TASK: getting all the stats(mean, std) of dataset to normalize, once each one is done save result in dataset_stats
    # clipart_stats = get_mean_std_dataset(args, 'clipart')                     #DONE
    # sketch_stats = get_mean_std_dataset(args, 'sketch')                       #DONE
    # painting_stats = get_mean_std_dataset(args, 'painting')                   #DATASET STILL NOT UPLOADED ON STORAGE
    # infograph_stats = get_mean_std_dataset(args, 'infograph')                 #DATASET STILL NOT UPLOADED ON STORAGE
    # quickdraw_stats = get_mean_std_dataset(args, 'quickdraw')                 #DONE
    # real_stats = get_mean_std_dataset(args, 'real')                           #DONE
 
    # ONGOING TASK: removing tasks containing <50 samples
    # source_ds = filter_classes(source_ds, logger)
    # one way is that once we figure out the classes to use, we remove the current dataset from storage and unzip only the chosen classes 
    # we would have to remake the stats but there will be no need to re-implement functions
    
    # Create Dataloaders and Split Train / Test
    # Question: in http://ai.bu.edu/M3SDA/ there is a train and test txt file, is that supposed to be used for the split? 
    # if it's the case if we remove some classes would that matter if we followed the split of the txt files anymore?
    # s_train, s_test = split_dl(args, source_ds, logger) 
    # t_train, t_test = split_dl(args, target_ds, logger)
    
    # #send data to device
    # s_train = DeviceDataLoader(s_train, args.device)
    # s_test = DeviceDataLoader(s_test, args.device)
    # t_train = DeviceDataLoader(t_train, args.device)
    # t_test = DeviceDataLoader(t_test, args.device)
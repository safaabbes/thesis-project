import os
import argparse

from torchvision import transforms 
from torchvision.datasets import ImageFolder
from collections import Counter

from utils import *

dataset_stats = {
  'clipart': ([0.7335894,0.71447897,0.6807669],[0.3542898,0.35537153,0.37871686]),
  'sketch': ([0.8326851 , 0.82697356, 0.8179188 ],[0.25409684, 0.2565908 , 0.26265645]),
  
}

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda', help='Computational Device')
    parser.add_argument('--path', type=str, default='/storage/TEV/sabbes/domainnet/', help='Dataset Path')
    parser.add_argument('--source_name', type=str, required=True, help='Source Domain Name')
    parser.add_argument('--target_name', type=str, required=True, help='Target Domain Name')
    parser.add_argument('--bs', type=int, default= 64, help='Batch Size')
    parser.add_argument('--n_epochs', type=int, default= 10, help='Number of Epochs')
    
    args=parser.parse_args()
    return args

def main():
    
    args = parse_args()
    
    logger = setup_logger('Logs/sample.log')
    
    for arg, value in vars(args).items():
        logger.info('{} = {}'.format(arg, value))
          
    source_transforms = [transforms.Resize((224, 224)),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize(*dataset_stats[args.source_name]),
                         ]
    target_transforms = [transforms.Resize((224, 224)),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(*dataset_stats[args.target_name]),
                        ]
    
    img_transform_source = transforms.Compose(source_transforms)
    img_transform_target = transforms.Compose(target_transforms)
    
    source_ds = ImageFolder(args.path+args.source_name, img_transform_source)
    target_ds = ImageFolder(args.path+args.target_name, img_transform_target)
    
    logger.info('number of {} samples: {}'.format(args.source_name , len(source_ds)))
    logger.info('number of {} samples: {}'.format(args.target_name , len(target_ds)))
    logger.info('Does the source and target have the same labels?: {}'.format(source_ds.classes == target_ds.classes))
    
    # ONGOING TASk: getting all the stats(mean, std) of dataset to normalize, once each one is done save result in dataset_stats
    # clipart_stats = get_mean_std_dataset(args, 'clipart')                     #DONE
    # sketch_stats = get_mean_std_dataset(args, 'sketch')                       #DONE
    # painting_stats = get_mean_std_dataset(args, 'painting')                   #DATASET STILL NOT UPLOADED
    # infograph_stats = get_mean_std_dataset(args, 'infograph')                 #DATASET STILL NOT UPLOADED
    # quickdraw_stats = get_mean_std_dataset(args, 'quickdraw')                 #SUBMITTED JOB
    # real_stats = get_mean_std_dataset(args, 'real')                           #SUBMITTED JOB
 
    # ONGOING TASk: removing tasks containing <50 samples
    # count = dict(Counter(source_ds.targets))
    # logger.info('Classes: {}'.format(count)) 
     

if __name__ == '__main__':
    main()
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
import sys

import torch
import torchvision
from sklearn.metrics import confusion_matrix
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image
import cv2
from PIL import Image, ImageDraw, ImageFont 

from datasets_biased import dataset2_biased as dataset
# from datasets import dataset2 as dataset
from models import resnet50s, resnet50s_1head
sys.path.append('..')
from utils import get_logger, deprocess


def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)

    # Test
    parser.add_argument('--bs', type=int, default=16)

    # Data
    parser.add_argument('--test_domain', type=str, required=True)
    parser.add_argument('--balance_mini_batches', default=False, action='store_true')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True)

    args = parser.parse_args()
    return args


def main():

    # Parse input arguments
    args_test = parse_args()

    # Update path to weights and runs
    args_test.path_weights = os.path.join('..','..', 'data', 'exps', 'biased_models', args_test.exp)

    # Load checkpoint
    checkpoint = torch.load(os.path.join(args_test.path_weights, '{:s}.tar'.format(args_test.checkpoint)))
    args = checkpoint['args']

    # Update training arguments
    args.exp = args_test.exp
    args.device = args_test.device
    args.num_workers = args_test.num_workers
    args.seed = args_test.seed
    args.bs = args_test.bs
    args.test_domain = args_test.test_domain
    args.checkpoint = args_test.checkpoint
    args.path_weights = args_test.path_weights

    # Create logger
    path_log = os.path.join(args.path_weights, 'viz_{:s}_{:s}.txt'.format(args.test_domain, args.checkpoint))
    logger = get_logger(path_log)

    # Activate CUDNN backend
    torch.backends.cudnn.enabled = True

    # Fix random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

    # Log input arguments
    for arg, value in vars(args).items():
        logger.info('{:s} = {:s}'.format(arg, str(value)))

    # Perform the test
    run_test(args, logger, checkpoint)


def run_test(args, logger, checkpoint):

    # Get the target dataset
    dataset_test = dataset(
        domain_type='{:s}_train'.format(args.test_domain),
        augm_type='train')

    # Get the source dataloaders
    if args.balance_mini_batches:
        weight_categories = 1.0 / torch.Tensor(dataset_test.instances1)
        weight_categories = weight_categories.double()
        weight_samples = np.array([weight_categories[_] for _ in dataset_test.labels1])
        weight_samples = torch.from_numpy(weight_samples)
        weight_samples = weight_samples.to(args.device)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_samples, len(dataset_test))
        loader_test = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=args.bs,
            num_workers=args.num_workers,
            sampler=sampler,
            shuffle=False,  # When using a custom sampler, we should turn off shuffling
            pin_memory=True,
            drop_last=True)
    else:
        loader_test = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=args.bs,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True)
    
    # Get model
    if args.model_type.lower() == 'resnet50s':
        model = resnet50s(args)
    elif args.model_type.lower() == 'resnet50s_1head':
        model = resnet50s_1head(args)
    else:
        raise NotImplementedError

    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # extract the weights of the classifier layer
    if args.model_type.lower() == 'resnet50s':
        head1 = model.head1
        head2 = model.head2 
        # convert the weights tensor to a numpy array
        weights_head1 = head1.weight.data.numpy()
        weights_head2 = head2.weight.data.numpy()
    elif args.model_type.lower() == 'resnet50s_1head':
        head = model.head
        weights_head = head.weight.data.numpy()
        
    
    feature_extractor = model.backbone
    
    # Send model to device
    feature_extractor.to(args.device)

    # logger.info('Learnable parameters:')
    # if hasattr(model, 'module'):
    #     params_to_update = model.module.parameters()
    #     for name, param in model.module.named_parameters():
    #         if param.requires_grad is True:
    #             logger.info(name)
    # else:
    #     params_to_update = model.parameters()
    #     for name, param in model.named_parameters():
    #         if param.requires_grad is True:
    #             logger.info(name)
                
    # logger.info('----------------------------------------------------------------------------')       
       
    # logger.info('feature_extractor Learnable parameters:')
    # if hasattr(feature_extractor, 'module'):
    #     params_to_update = feature_extractor.module.parameters()
    #     for name, param in feature_extractor.module.named_parameters():
    #         if param.requires_grad is True:
    #             logger.info(name)
    # else:
    #     params_to_update = feature_extractor.parameters()
    #     for name, param in feature_extractor.named_parameters():
    #         if param.requires_grad is True:
    #             logger.info(name)
    
    
    # Put model in evaluation mode
    feature_extractor.eval()   
    
    features_list = list()
    labels_list = list()
 
    # Loop over test mini-batches
    for i, data in enumerate(loader_test):

        # Load mini-batch
        images, categories1, categories2 = data
        images = images.to(args.device, non_blocking=True)
        categories1 = categories1.to(args.device, non_blocking=True)
        categories2 = categories2.to(args.device, non_blocking=True)

        with torch.inference_mode():

            # Forward pass
            features = feature_extractor(images)
            
            # features = torch.nn.functional.normalize(features)
            
            features_list.extend(features.tolist())
            
            labels_list.extend(zip(categories1.tolist(),categories2.tolist()))
            
            if len(set(labels_list)) == 40 and len(labels_list) > 1500:
                break 
    
    print(len(labels_list))
    print(len(features_list))
    print('WARNING, DATASET IF GIVING THE WRONG CLUSTER LABEL, TO TEST WITH CORRECT LABELS IMPORT THE CORRECT DATASET!')

    if args.model_type.lower() == 'resnet50s':
        np.savez(os.path.join(args.path_weights,'F_2H_{}_bias.npz'.format(args.test_domain)), 
                weights_head1= weights_head1, 
                weights_head2= weights_head2,
                features= np.array(features_list),
                labels= labels_list,
                )
                    
    elif args.model_type.lower() == 'resnet50s_1head':
        np.savez(os.path.join(args.path_weights,'F_1H_{}_bias.npz'.format(args.test_domain)), 
                weights_head= weights_head, 
                features= np.array(features_list),
                labels= labels_list,
                )
                    
                     
if __name__ == '__main__':
    main()

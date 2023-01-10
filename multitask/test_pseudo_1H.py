import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
import sys

import torch
import torchvision

from datasets import dataset2 as dataset
from datasets import PseudoLabelDataset
from models import resnet50s_1head
sys.path.append('..')
from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)

    # Test
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--condition', type=str, required=True)

    # Data
    parser.add_argument('--target_train', type=str, required=True)

    # Model
    parser.add_argument('--checkpoint', type=str, required=True)

    args = parser.parse_args()
    return args


def main():

    # Parse input arguments
    args_test = parse_args()

    # Update path to weights and runs
    args_test.path_weights = os.path.join('..','..', 'data', 'exps', 'weights', args_test.exp)
    args_test.path_pseudo = os.path.join('..','..','data', 'exps', 'pseudo', args_test.exp)
    
    # Create pseudo exp folder
    os.makedirs(args_test.path_pseudo, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(os.path.join(args_test.path_weights, '{:s}.tar'.format(args_test.checkpoint)))
    args = checkpoint['args']

    # Update training arguments
    args.exp = args_test.exp
    args.device = args_test.device
    args.num_workers = args_test.num_workers
    args.seed = args_test.seed
    args.bs = args_test.bs
    args.target_train = args_test.target_train
    args.checkpoint = args_test.checkpoint
    args.path_weights = args_test.path_weights
    args.path_pseudo = args_test.path_pseudo
    args.condition = args_test.condition


    # Create logger
    path_log = os.path.join(args_test.path_pseudo, 'log_test_{:s}_{:s}_{:s}.txt'.format(args.target_train, args.checkpoint, args.condition))
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
        domain_type=args.target_train,
        augm_type='train')
    logger.info('Train samples: {:d}'.format(len(dataset_test)))
    
    # Get the test dataloader
    loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    # Get model
    if args.model_type.lower() == 'resnet50s_1head':
        model = resnet50s_1head(args)
    else:
        raise NotImplementedError

    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    # Send model to device
    model.to(args.device)
    # logger.info('Model is on device: {}'.format(next(model.parameters()).device))

    # Put model in evaluation mode
    model.eval()
    
    pseudo_labels_1 = torch.zeros(len(dataset_test), dtype=torch.long).to(args.device)
    pseudo_labels_2 = torch.zeros(len(dataset_test), dtype=torch.long).to(args.device)

    # Init stats
    running_oa1, running_mca1_num, running_mca1_den = list(), list(), list()
    running_oa2, running_mca2_num, running_mca2_den = list(), list(), list()
    ground_truth_1, ground_truth_2 = list(), list()


    # Determine the size and data type of the images and labels
    image_size = (3, 224, 224)  
    image_dtype = torch.float  
    label_dtype = torch.long  
    max_num_images = 17000
    pseudo_images = torch.empty((max_num_images, *image_size), dtype=image_dtype)
    pseudo_labels1 = torch.empty(max_num_images, dtype=label_dtype)
    pseudo_labels2 = torch.empty(max_num_images, dtype=label_dtype)
    num_images = 0
    num_labels1 = 0
    num_labels2 = 0
    
    # Loop over test mini-batches
    for i, data in enumerate(loader_test):

        # Load mini-batch
        images, categories1, categories2 = data
        images = images.to(args.device, non_blocking=True)
        categories1 = categories1.to(args.device, non_blocking=True)
        categories2 = categories2.to(args.device, non_blocking=True)
        ground_truth_1.extend(categories1.tolist())
        ground_truth_2.extend(categories2.tolist())
        with torch.inference_mode():

            # Forward pass
            logits1 = model(images)
            
            tmp = np.load('mapping.npz')
            mapping = torch.tensor(tmp['data'], dtype=torch.float32, device=args.device, requires_grad=False)
            logits2 = torch.mm(logits1, mapping) / (1e-6 + torch.sum(mapping, dim=0))
            
            # TODO apply softmax before deducing logit2
            logits1 = torch.softmax(logits1, dim=1)
            prob1, preds1 = torch.max(logits1, dim=1)
            
            logits2 = torch.softmax(logits2, dim=1)
            prob2, preds2 = torch.max(logits2, dim=1)
            
            # Creating Super Classes based on the prediction of the super-classes if they surpass th=0.9
            if args.condition == '0.9' or args.condition == '0.8':
                threshold = float(args.condition)
                for j, p in enumerate(prob2):
                    if p.item() > threshold:
                        pseudo_images[num_images] = images[j]
                        pseudo_labels1[num_labels1] = preds1[j]
                        pseudo_labels2[num_labels2] = preds2[j]
                        num_images += 1
                        num_labels1 += 1
                        num_labels2 += 1
                # Creating Pseudo Labels based on the probability confidence using a threshold 
                mask = prob1 > threshold
                pseudo_labels_1[i*args.bs : (i+1)*args.bs][mask] = preds1[mask]
                pseudo_labels_1[i*args.bs : (i+1)*args.bs][~mask] = -1  
                mask = prob2 > threshold
                pseudo_labels_2[i*args.bs : (i+1)*args.bs][mask] = preds2[mask]
                pseudo_labels_2[i*args.bs : (i+1)*args.bs][~mask] = -1  
            # Creating Super Classes based on the coherence between class and super-class           
            elif args.condition == 'coherence':
                one_hot_preds1 = torch.nn.functional.one_hot(preds1, num_classes=args.num_categories1).to(torch.float)
                one_hot_preds2 = torch.nn.functional.one_hot(preds2, num_classes=args.num_categories2).to(torch.float)
                preds1_true_sc = torch.mm(one_hot_preds1, mapping)
                mask = (preds1_true_sc == one_hot_preds2).all(dim=1)
                for j, p in enumerate(mask):
                    if p.item():
                        pseudo_images[num_images] = images[j]
                        pseudo_labels1[num_labels1] = preds1[j]
                        pseudo_labels2[num_labels2] = preds2[j]
                        num_images += 1
                        num_labels1 += 1
                        num_labels2 += 1
                # Creating Pseudo Labels if the class prediction belongs to the cluster prediction
                pseudo_labels_1[i*args.bs : (i+1)*args.bs][mask] = preds1[mask]
                pseudo_labels_1[i*args.bs : (i+1)*args.bs][~mask] = -1
                pseudo_labels_2[i*args.bs : (i+1)*args.bs][mask] = preds2[mask]
                pseudo_labels_2[i*args.bs : (i+1)*args.bs][~mask] = -1
            
        # Update metrics
        oa1 = torch.sum(preds1 == categories1.squeeze()) / len(categories1)
        running_oa1.append(oa1.item())
        mca1_num = torch.sum(
            torch.nn.functional.one_hot(preds1, num_classes=args.num_categories1) * \
            torch.nn.functional.one_hot(categories1, num_classes=args.num_categories1), dim=0)
        mca1_den = torch.sum(
            torch.nn.functional.one_hot(categories1, num_classes=args.num_categories1), dim=0)
        running_mca1_num.append(mca1_num.detach().cpu().numpy())
        running_mca1_den.append(mca1_den.detach().cpu().numpy())

        oa2 = torch.sum(preds2 == categories2.squeeze()) / len(categories2)
        running_oa2.append(oa2.item())
        mca2_num = torch.sum(
            torch.nn.functional.one_hot(preds2, num_classes=args.num_categories2) * \
            torch.nn.functional.one_hot(categories2, num_classes=args.num_categories2), dim=0)
        mca2_den = torch.sum(
            torch.nn.functional.one_hot(categories2, num_classes=args.num_categories2), dim=0)
        running_mca2_num.append(mca2_num.detach().cpu().numpy())
        running_mca2_den.append(mca2_den.detach().cpu().numpy())
    
    logger.info('Pseudo Labels Strategy: {}'.format(args.condition))    
     
    logger.info('Pseudo Labels Analysis for Original Classes')
    pseudo_labels_1 = pseudo_labels_1.cpu().numpy()
    discarded_labels = np.sum(pseudo_labels_1 == -1)
    filtered_pseudo_labels_1 = len(pseudo_labels_1) - discarded_labels
    count = 0
    for i in range(len(pseudo_labels_1)):
        if pseudo_labels_1[i] == -1:
            continue
        if pseudo_labels_1[i] == ground_truth_1[i]:
            count += 1
    logger.info('\t Number of Pseudo Labels =  {}'.format(filtered_pseudo_labels_1))
    logger.info('\t Number of actually correct Pseudo Labels =  {}'.format(count))
    logger.info('\t Number of Erroneous Pseudo Labels =  {}'.format(filtered_pseudo_labels_1-count))
    
    logger.info('Pseudo Labels Analysis for Super Classes')
    pseudo_labels_2 = pseudo_labels_2.cpu().numpy()
    discarded_labels = np.sum(pseudo_labels_2 == -1)
    filtered_pseudo_labels_2 = len(pseudo_labels_2) - discarded_labels
    count = 0
    for i in range(len(pseudo_labels_2)):
        if pseudo_labels_2[i] == -1:
            continue
        if pseudo_labels_2[i] == ground_truth_2[i]:
            count += 1
    logger.info('\t Number of Pseudo Labels =  {}'.format(filtered_pseudo_labels_2))
    logger.info('\t Number of actually correct Pseudo Labels =  {}'.format(count))
    logger.info('\t Number of Erroneous Pseudo Labels =  {}'.format(filtered_pseudo_labels_2-count))
    
    pseudo_images = pseudo_images[:num_images]
    pseudo_labels1 = pseudo_labels1[:num_labels1]
    pseudo_labels2 = pseudo_labels2[:num_labels2]
    
    torch.save({'images': pseudo_images, 
                'pseudo_labels1': pseudo_labels1,
                'pseudo_labels2': pseudo_labels2,
                'pseudo_domain': args.target_train,
                'source_exp': args.exp,
                'checkpoint': args.checkpoint,
                'condition': args.condition,
                }, os.path.join(args.path_pseudo, '{}_{}_{}.tar'.format(args.target_train, args.checkpoint, args.condition)))
    
    logger.info('\t images len =  {}'.format(len(pseudo_images)))
    logger.info('\t labels1 len =  {}'.format(len(pseudo_labels1)))
    logger.info('\t labels2 len =  {}'.format(len(pseudo_labels2)))
    logger.info('labels1 =  {}'.format(pseudo_labels1[0:40]))
    logger.info('labels2 =  {}'.format(pseudo_labels2[0:40]))
       
    # Update MCA metric
    mca1_num = np.sum(running_mca1_num, axis=0)
    mca1_den = 1e-16 + np.sum(running_mca1_den, axis=0)
    mca2_num = np.sum(running_mca2_num, axis=0)
    mca2_den = 1e-16 + np.sum(running_mca2_den, axis=0)

    stats = {
        'oa1': np.mean(running_oa1)*100,
        'mca1': np.mean(mca1_num/mca1_den)*100,
        'oa2': np.mean(running_oa2)*100,
        'mca2': np.mean(mca2_num/mca2_den)*100,
        }

    logger.info('OA1: {:.4f}, MCA1: {:.4f}, OA2: {:.4f}, MCA2: {:.4f}'.format(
        stats['oa1'], stats['mca1'], stats['oa2'], stats['mca2']))
    
if __name__ == '__main__':
    main()
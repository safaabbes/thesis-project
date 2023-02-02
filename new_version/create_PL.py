import argparse
import numpy as np
import os
import random
import sys

import torch
import torchvision
import torch.nn.functional as F

from datasets import dataset
from models import resnet50_1h, resnet50_2h
sys.path.append('..')
from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--train_seed', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)

    # Test
    parser.add_argument('--bs', type=int, default=16)
    
    # Data
    parser.add_argument('--PL_domain', type=str, required=True)
    parser.add_argument('--condition', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=0.9)

    # Model
    parser.add_argument('--checkpoint', type=str, required=True)

    args = parser.parse_args()
    return args


def main():

    # Parse input arguments
    args_test = parse_args()

    # Update path to weights and runs
    args_test.path_weights = os.path.join('..','..', 'data', 'new_exps', args_test.arch, args_test.exp, args_test.train_seed)

    # Load checkpoint
    checkpoint = torch.load(os.path.join(args_test.path_weights, '{:s}.tar'.format(args_test.checkpoint)))
    args = checkpoint['args']

    # Update training arguments
    args.exp = args_test.exp
    args.device = args_test.device
    args.num_workers = args_test.num_workers
    args.seed = args_test.seed
    args.bs = args_test.bs
    args.PL_domain = args_test.PL_domain
    args.checkpoint = args_test.checkpoint
    args.path_weights = args_test.path_weights
    args.condition = args_test.condition
    args.threshold = args_test.threshold

    # Create logger
    path_log = os.path.join(args.path_weights, 'log_PL_{:s}_{:s}.txt'.format(args.PL_domain, args.checkpoint))
    logger = get_logger(path_log)

    # Activate CUDNN backend
    torch.backends.cudnn.enabled = False

    # Log input arguments
    for arg, value in vars(args).items():
        logger.info('{:s} = {:s}'.format(arg, str(value)))
        
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

    # Perform the test
    run_create_PL(args, logger, checkpoint)


def run_create_PL(args, logger, checkpoint):

    # Get the target dataset
    dataset_test = dataset(
        domain_type='{:s}_train'.format(args.PL_domain),
        augm_type='test')
    logger.info('Total PL candidates: {:d}'.format(len(dataset_test)))
    
    # Get the test dataloader
    loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=False, #DO NOT SHUFFLE OR WE GET WRONG ASSIGNMENTS OF THE FILES
        pin_memory=True,
        drop_last=False)

    # Get model
    if args.model_type.lower() == 'resnet50_1h':
        model = resnet50_1h(args)
    elif args.model_type.lower() == 'resnet50_2h':
        model = resnet50_2h(args)
    else:
        raise NotImplementedError

    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    # Send model to device
    model.to(args.device)
    # logger.info('Model is on device: {}'.format(next(model.parameters()).device))

    # Put model in evaluation mode
    model.eval()

    # Init stats
    file_names = loader_test.dataset.pointer
    pseudo_file_names = list()
    correct_class = 0
    correct_sc = 0
    # Loop over test mini-batches
    for i, data in enumerate(loader_test):
        # Get filenames for the batch
        start = i * args.bs % len(file_names)
        end = start + args.bs
        images_file_names = file_names[start:end]
        
        # Load mini-batch
        images, categories1, categories2 = data
        images = images.to(args.device, non_blocking=True)
        categories1 = categories1.to(args.device, non_blocking=True)
        categories2 = categories2.to(args.device, non_blocking=True)

        with torch.inference_mode():
            # Forward pass
            if args.model_type.lower() == 'resnet50_1h':
                logits1 = model(images)
                tmp = np.load('mapping.npz')
                mapping = torch.tensor(tmp['data'], dtype=torch.float32, device=args.device, requires_grad=False)
                logits2 = torch.mm(logits1, mapping) / (1e-6 + torch.sum(mapping, dim=0))
            elif args.model_type.lower() == 'resnet50_2h': 
                logits1, logits2 = model(images)
            else:
                NotImplementedError
        
        # logger.info('logits1 {}'.format(logits1))
        logits1 = F.softmax(logits1, dim=-1)
        # logger.info('logits1 {}'.format(logits1))
        logits2 = F.softmax(logits2, dim=-1)
        probs1, preds1 = torch.max(logits1, dim=1)
        probs2, preds2 = torch.max(logits2, dim=1)
        
        # Get Pseudo Labels
        # Confidence thresholding is a must because coherence alone doesn't create reliable PL however there are two options to thresholding: on class prob or on sc_prob 
        # Getting Coherence
        one_hot_preds1 = torch.nn.functional.one_hot(preds1, num_classes=args.num_categories1).to(torch.float)
        one_hot_preds2 = torch.nn.functional.one_hot(preds2, num_classes=args.num_categories2).to(torch.float)
        tmp = np.load('mapping.npz')
        mapping = torch.tensor(tmp['data'], dtype=torch.float32, device=args.device, requires_grad=False)
        preds1_true_sc = torch.mm(one_hot_preds1, mapping)
        mask_coherence = (preds1_true_sc == one_hot_preds2).all(dim=1)
        for j, p in enumerate(mask_coherence):
            threshold = args.threshold
            if args.condition == 'confidence_on_class':
                # logger.info('categories1 {}'.format(categories1))
                # logger.info(' preds1 {}'.format(preds1))
                # logger.info(' probs1 {}'.format(probs1))
                if p.item() and probs1[j]>threshold:
                    pseudo_file_names.append([str(images_file_names[j][0]), str(preds1[j].cpu().tolist()), str(preds2[j].cpu().tolist())])
                    # Correct PL Check
                    if preds1[j] == categories1[j]:
                        correct_class += 1
                    if preds2[j] == categories2[j]:
                        correct_sc += 1
            elif args.condition == 'confidence_on_sc':
                if p.item() and probs2[j]>threshold:
                    pseudo_file_names.append([str(images_file_names[j][0]), str(preds1[j].cpu().tolist()), str(preds2[j].cpu().tolist())])
                    # Correct PL Check
                    if preds1[j] == categories1[j]:
                        correct_class += 1
                    if preds2[j] == categories2[j]:
                        correct_sc += 1

    stats = {
        'all_PL': len(pseudo_file_names),
        'Total_%': len(pseudo_file_names)/len(dataset_test)*100,
        'correct_class': correct_class,
        'wrong_class': len(pseudo_file_names) - correct_class,
        'correct_sc': correct_sc,
        'correct_%': correct_sc/len(pseudo_file_names)*100,
        'wrong_sc': len(pseudo_file_names) - correct_sc,
        }

    logger.info('Total PL: {}, Total_%: {:.2f}, correct_%: {:.2f}, \n \
                PL with correct_class: {}, wrong_class: {}, \n \
                PL with correct_sc: {}, wrong_sc: {}'.format(
        stats['all_PL'], stats['Total_%'], stats['correct_%'], 
        stats['correct_class'], stats['wrong_class'], 
        stats['correct_sc'],  stats['wrong_sc']))
    
    # logger.info('{}'.format(pseudo_file_names[:30]))
    
    torch.save({'images_file_array': pseudo_file_names, 
                'PL_domain': args.PL_domain,
                'used_exp': args.exp,
                'checkpoint': args.checkpoint,
                'condition': args.condition,
                'threshold': args.threshold,
                }, os.path.join(args.path_weights, '{}_{}_{}_{}.tar'.format(args.PL_domain, args.checkpoint, args.threshold, args.condition)))

if __name__ == '__main__':
    main()
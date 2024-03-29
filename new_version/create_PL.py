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
    parser.add_argument('--condition', type=str, default='on_class')
    parser.add_argument('--percentage', type=float, default=0.5)

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
    args.percentage = args_test.percentage

    # Create logger
    path_log = os.path.join(args.path_weights, 'LOG_PL_{}_{}_{}_{}.txt'.format(args.PL_domain, args.checkpoint, args.percentage, args.condition))
    logger = get_logger(path_log)

    # Activate CUDNN backend
    torch.backends.cudnn.enabled = True

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
    pseudo_pointer = list()
    # pseudo_file_names = list()
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

        for j in range(len(images_file_names)):
            pseudo_pointer.append([str(images_file_names[j][0]), 
                                    str(categories1[j].item()), 
                                    str(preds1[j].item()),
                                    str(probs1[j].item()), 
                                    str(categories2[j].item()), 
                                    str(preds2[j].item()),
                                    str(probs2[j].item()),
            ])

    # Sort All Samples decreasingly along the probabilities
    if args.condition == 'on_class':
        sorted_pseudo_pointer = sorted(pseudo_pointer, key=lambda x: x[3], reverse=True)
    elif args.condition == 'on_sc':
        sorted_pseudo_pointer = sorted(pseudo_pointer, key=lambda x: x[6], reverse=True)
    else:
        NotImplementedError
    # Select highest % of confident PL
    selected_nb = int(args.percentage * len(sorted_pseudo_pointer))
    selected_PL = np.array(sorted_pseudo_pointer[:selected_nb])
    

    # PL Statistics
    # Nb of Correct Classes
    correct_categories = np.sum(np.equal(selected_PL[:, 1].astype(int), selected_PL[:, 2].astype(int)))
    correct_categories_P = correct_categories / selected_nb 
    # Nb of Corrent Super-Classes
    correct_sc = np.sum(np.equal(selected_PL[:, 4].astype(int), selected_PL[:, 5].astype(int)))
    correct_sc_P = correct_sc / selected_nb 
    # Nb of Coherent assignments
    # coherent_PL =
    categories_preds = torch.from_numpy(selected_PL[:, 2].astype(int))
    categories_preds = torch.nn.functional.one_hot(categories_preds, num_classes=args.num_categories1).to(args.device).to(torch.float)
    tmp = np.load('mapping.npz')
    mapping = torch.tensor(tmp['data'], dtype=torch.float32, device=args.device, requires_grad=False)
    categories_true_sc = torch.mm(categories_preds, mapping)
    sc_preds = torch.from_numpy(selected_PL[:, 4].astype(int))
    sc_preds = torch.nn.functional.one_hot(sc_preds, num_classes=args.num_categories2).to(args.device).to(torch.float)
    nb_coherence = torch.sum((categories_true_sc == sc_preds).all(dim=1)).item()
    coherence_P = nb_coherence / selected_nb

    stats = {
        'selected_nb': selected_nb,
        'correct_categories': correct_categories,
        'correct_categories_P': correct_categories_P,
        'correct_sc': correct_sc,
        'correct_sc_P': correct_sc_P,
        'nb_coherence': nb_coherence,
        'coherence_P': coherence_P,
        }

    logger.info('Nb Selected PL: {}, \n \
        Nb correct categories: {}, Correct Categories %: {:.4f} \n \
        Nb correct SC: {}, Correct SC %: {:.4f} \n \
        Nb of Coherent Class-SC: {}, Coherence %: {:.4f}'.format(   
        stats['selected_nb'], stats['correct_categories'], stats['correct_categories_P'], 
        stats['correct_sc'], stats['correct_sc_P'], stats['nb_coherence'],  stats['coherence_P']))
    

    torch.save({'selected_PL': selected_PL, 
                'PL_domain': args.PL_domain,
                'used_exp': args.exp,
                'checkpoint': args.checkpoint,
                'percentage': args.percentage,
                'condition': args.condition,
                }, os.path.join(args.path_weights, '{}_{}_{}_{}.tar'.format(args.PL_domain, args.checkpoint, args.percentage, args.condition)))

if __name__ == '__main__':
    main()
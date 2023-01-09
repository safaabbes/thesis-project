import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
import sys

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from datasets import dataset2 as dataset
from models import resnet50a, resnet50b, resnet50c, resnet50d, resnet50e, resnet50s
sys.path.append('..')
from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=None)

    # Test
    parser.add_argument('--bs', type=int, default=16)

    # Data
    parser.add_argument('--test_domain', type=str, required=True)

    # Model
    parser.add_argument('--checkpoint', type=str, required=True)

    args = parser.parse_args()
    return args


def main():

    # Parse input arguments
    args_test = parse_args()

    # Update path to weights and runs
    args_test.path_weights = os.path.join('..', 'data', 'exps', 'weights', args_test.exp)
    args_test.path_runs = os.path.join('..', 'data', 'exps', 'runs', args_test.exp)

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
    args.path_runs = args_test.path_runs

    # Create logger
    path_log = os.path.join(args.path_weights, 'log_test_{:s}.txt'.format(args.test_domain))
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
        path_pointer=os.path.join('..', 'data', 'splits_multitask', '{:s}_test_mini.txt'.format(args.test_domain)),
        augm_type='test')
    logger.info('Test samples: {:d}'.format(len(dataset_test)))
    
    # Get the test dataloader
    loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    # Get model
    if args.model_type.lower() == 'resnet50a':
        model = resnet50a(args)
    elif args.model_type.lower() == 'resnet50b':
        model = resnet50b(args)
    elif args.model_type.lower() == 'resnet50c':
        model = resnet50c(args)
    elif args.model_type.lower() == 'resnet50d':
        model = resnet50d(args)
    elif args.model_type.lower() == 'resnet50e':
        model = resnet50e(args)
    elif args.model_type.lower() == 'resnet50s':
        model = resnet50s(args)
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
    running_oa1, running_mca1_num, running_mca1_den = list(), list(), list()
    running_oa2, running_mca2_num, running_mca2_den = list(), list(), list()
    # Loop over test mini-batches
    for data in loader_test:

        # Load mini-batch
        images, categories1, categories2 = data
        images = images.to(args.device, non_blocking=True)
        categories1 = categories1.to(args.device, non_blocking=True)
        categories2 = categories2.to(args.device, non_blocking=True)

        with torch.inference_mode():

            # Forward pass
            logits1, logits2 = model(images)
            _, preds1 = torch.max(logits1, dim=1)
            _, preds2 = torch.max(logits2, dim=1)

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

    # Confusion matrix
    # tmp = np.asarray(tmp)
    # cm = np.zeros((args.num_categories, args.num_categories), dtype=float)
    # for i, j in zip(tmp[:, 1], tmp[:, 0]):
    #     cm[i, j] += 1
    # unique, counts = np.unique(tmp[:, 1], return_counts=True)
    # for _ in range(args.num_categories):
    #     cm[_, :] /= 1.0 * counts[_] + np.finfo(float).eps

    # plt.figure()
    # sns.set(font_scale=0.5)
    # ax = sns.heatmap(cm,
    #     cmap='Blues', cbar=False, vmin=0.0, vmax=1.0,
    #     annot=True, annot_kws={'fontsize': 3}, fmt='.2f',
    #     xticklabels=loader_test.dataset.category_names, yticklabels=loader_test.dataset.category_names)  # mask=cm==0.0
    # ax.tick_params(left=False, bottom=False)
    # ax.set_xlabel('Predicted classes')
    # ax.set_ylabel('Ground-truth classes')
    # ax.title.set_text('OA: {:.4f}, MCA: {:.4f}'.format(stats['oa'], stats['mca']))
    # plt.savefig(os.path.join(args.path_weights, 'cm1_{:s}.png'.format(args.test_domain)),
    #     transparent=False, bbox_inches='tight', dpi=300)
    # plt.close()


if __name__ == '__main__':
    main()


# if args.path_pointers[-2:] == '28':
#     C = 28
# else:
#     C = 30
# cm = torch.zeros(C, C, dtype=torch.int, device=args.device)
# for i, j in zip(labels_source, preds):
#     cm[i, j] += 1
# cm = np.array(cm.detach().cpu().numpy(), dtype=float)
# unique, counts = np.unique(labels_source, return_counts=True)
# for _ in range(C):
#     cm[_, :] /= 1.0 * counts[_] + np.finfo(float).eps

# plt.figure()
# sns.set(font_scale=0.5)
# ax = sns.heatmap(cm,
#     cmap='Blues', cbar=False, vmin=0.0, vmax=1.0,
#     annot=True, annot_kws={'fontsize': 4}, fmt='.2f',
#     xticklabels=loader_source.dataset.category_names, yticklabels=loader_source.dataset.category_names)  # mask=cm==0.0
# ax.tick_params(left=False, bottom=False)
# ax.set_xlabel('Predicted classes')
# ax.set_ylabel('Ground-truth classes')
# plt.savefig(os.path.join(args.path_weights, '{:s}_cm_{:s}.png'.format(args.checkpoint, name_source)),
#     transparent=False, bbox_inches='tight', dpi=300)
# plt.close()
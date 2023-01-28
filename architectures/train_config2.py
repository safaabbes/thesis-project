'''
This is the Source-Only Multitask Configuration where we use a source and its correct super-classes
'''
import argparse
from itertools import cycle
import numpy as np
import os
import random
import sys
import time

import torch
import torchvision
import torchinfo
import pytorch_warmup as warmup

from datasets import dataset_2 as dataset
from models import resnet50_1head
from losses import loss_ce
sys.path.append('..')
from utils import get_logger

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--config', type=str, default='config2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--freq_saving', type=int, default=10)

    # Train
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--balance_mini_batches', default=False, action='store_true')

    # Data
    parser.add_argument('--source_train', type=str, required=True)
    parser.add_argument('--source_test', type=str, required=True)

    # Model
    parser.add_argument('--model_type', type=str, required=True)

    # Optimizer
    parser.add_argument('--optim_type', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=1e-02)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--w_decay', type=float, default=1e-05)
    
    # Losses
    parser.add_argument('--mu1', type=float, default=0.5)
    parser.add_argument('--mu2', type=float, default=0.5)

    args = parser.parse_args()
    return args


def main():

    # Parse input arguments
    args = parse_args()

    # Update path to weights and runs
    args.path_weights = os.path.join('..', '..','data', 'experiments', args.config, args.exp)

    # Create experiment folder
    os.makedirs(args.path_weights, exist_ok=True)

    # Create logger
    logger = get_logger(os.path.join(args.path_weights, 'log_train.txt'))

    # Log library versions
    logger.info('PyTorch version = {:s}'.format(torch.__version__))
    logger.info('TorchVision version = {:s}'.format(torchvision.__version__))

    # Activate CUDNN backend
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
    for arg, value in vars(args).items():
        logger.info('{:s} = {:s}'.format(arg, str(value)))

    # Perform the training
    run_train(args, logger)


def run_train(args, logger):

    # Get the source datasets
    dataset_train_source = dataset(
        domain_type=args.source_train,
        augm_type='train')
    dataset_valid_source = dataset(
        domain_type=args.source_test,
        augm_type='test')

    # Log stats
    logger.info('Source samples, Training: {:d}, Validation: {:d}'.format(
        len(dataset_train_source), len(dataset_valid_source)))

    # Get the source dataloaders
    if args.balance_mini_batches:
        weight_categories = 1.0 / torch.Tensor(dataset_train_source.instances1)
        weight_categories = weight_categories.double()
        weight_samples = np.array([weight_categories[_] for _ in dataset_train_source.labels1])
        weight_samples = torch.from_numpy(weight_samples)
        weight_samples = weight_samples.to(args.device)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_samples, len(dataset_train_source))
        loader_train_source = torch.utils.data.DataLoader(
            dataset=dataset_train_source,
            batch_size=args.bs,
            num_workers=args.num_workers,
            sampler=sampler,
            shuffle=False,  # When using a custom sampler, we should turn off shuffling
            pin_memory=True,
            drop_last=True)
    else:
        loader_train_source = torch.utils.data.DataLoader(
            dataset=dataset_train_source,
            batch_size=args.bs,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True)
    loader_valid_source = torch.utils.data.DataLoader(
        dataset=dataset_valid_source,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    # Get the model
    if args.model_type.lower() == 'resnet50_1head':
        model = resnet50_1head()
    else:
        raise NotImplementedError

    # Send the model to the device
    model = model.to(args.device)

    # Set data parallelism
    if torch.cuda.device_count() == 1:
        logger.info('Using a single GPU, data parallelism is disabled')
    else:
        logger.info('Using multiple GPUs, with data parallelism')
        model = torch.nn.DataParallel(model)

    # Set the model in training mode
    model.train()

    head = ['head.weight', 'head.bias']
    params_head = list(filter(lambda kv: kv[0] in head, model.named_parameters()))
    params_back = list(filter(lambda kv: kv[0] not in head, model.named_parameters()))

    # Get the optimizer
    if args.optim_type == 'SGD':
        optimizer = torch.optim.SGD(
            [
                {'params': [p for n, p in params_back], 'lr': 0.1 * args.lr},
                {'params': [p for n, p in params_head], 'lr': args.lr}
            ],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.w_decay,
            nesterov=True)
    else:
        raise NotImplementedError

    # Get losses and send them to the device
    criterion1 = loss_ce()
    criterion1 = criterion1.to(args.device)

    # Loop over epochs
    start = time.time()
    for epoch in range(1, args.num_epochs + 1):
        
        # Training
        since = time.time()
        stats_train = do_epoch_train(loader_train_source, model, criterion1, optimizer, args, logger)
        logger.info('TRN, Epoch: {:4d}, Loss: {:e}, OA1: {:.4f}, MCA1: {:.4f}, Elapsed: {:.1f}s'.format(
            epoch, stats_train['loss'], stats_train['oa1'], stats_train['mca1'], time.time() - since))
        
        # Validation
        since = time.time()
        stats_valid = do_epoch_valid(loader_valid_source, model, criterion1, args)
        logger.info('VAL, Epoch: {:4d}, Loss: {:e}, OA1: {:.4f}, MCA1: {:.4f}, Elapsed: {:.1f}s'.format(
            epoch, stats_valid['loss'], stats_valid['oa1'], stats_valid['mca1'], time.time() - since))

        # Save current checkpoint
        if epoch % args.freq_saving == 0:
            torch.save({
                'epoch': epoch,
                'args': args,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()},
                os.path.join(args.path_weights, '{:04d}.tar'.format(epoch)))

    # Save last checkpoint
    torch.save({
        'epoch': epoch,
        'args': args,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()},
        os.path.join(args.path_weights, 'last.tar'))

    end = time.time()
    logger.info('Elapsed time: {:.2f} minutes'.format((end - start)/60))


def do_epoch_train(loader_train_source, model, criterion1, optimizer, args, logger):

    # Set the model in training mode
    model = model.train()

    # Init stats
    running_loss = list()
    running_oa1, running_mca1_num, running_mca1_den = list(), list(), list()

    for i, data in enumerate(loader_train_source):

        # Load source mini-batch
        images, labels, correct_sc = data
        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        correct_sc = correct_sc.to(args.device, non_blocking=True)

        # Zero the parameters gradients
        optimizer.zero_grad()

        # Forward pass for source data
        logits1 = model(images)
        _, preds1 = torch.max(logits1, dim=1)
        
        # Correct Super-Class logits
        tmp = np.load('correct_mapping.npz', allow_pickle=True)
        mapping = torch.tensor(tmp['data'], dtype=torch.float32, device=args.device, requires_grad=False)
        logits2 = torch.mm(logits1, mapping) / (1e-6 + torch.sum(mapping, dim=0))

        # Losses
        source_loss1 = args.mu1 * criterion1(logits1, labels)
        source_loss2 = args.mu2 * criterion1(logits2, correct_sc)
        loss = source_loss1 + source_loss2

        # Back-propagation
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Update losses
        running_loss.append(loss.item())

        # Update metrics
        oa1 = torch.sum(preds1 == labels.squeeze()) / len(labels)
        running_oa1.append(oa1.item())
        mca1_num = torch.sum(
            torch.nn.functional.one_hot(preds1, num_classes=40) * \
            torch.nn.functional.one_hot(labels, num_classes=40), dim=0)
        mca1_den = torch.sum(
            torch.nn.functional.one_hot(labels, num_classes=40), dim=0)
        running_mca1_num.append(mca1_num.detach().cpu().numpy())
        running_mca1_den.append(mca1_den.detach().cpu().numpy())

    # Update MCA metric
    mca1_num = np.sum(running_mca1_num, axis=0)
    mca1_den = 1e-16 + np.sum(running_mca1_den, axis=0)

    stats = {
        'loss': np.mean(running_loss),
        'oa1': np.mean(running_oa1),
        'mca1': np.mean(mca1_num/mca1_den),
        }

    return stats

def do_epoch_valid(loader_valid_source, model, criterion1, args):

    # Set the model in evaluation mode
    model = model.eval()

    # Init stats
    running_loss = list()
    running_oa1, running_mca1_num, running_mca1_den = list(), list(), list()

    # Loop over validation mini-batches
    for data in loader_valid_source:

        # Load source mini-batch
        images, labels, _ = data
        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)

        with torch.inference_mode():

            # Forward pass for source data
            logits = model(images)
            _, preds = torch.max(logits, dim=1)

            # Loss
            loss = criterion1(logits, labels)

        # Update loss
        running_loss.append(loss.item())

        # Update metrics
        oa1 = torch.sum(preds == labels.squeeze()) / len(labels)
        running_oa1.append(oa1.item())
        mca1_num = torch.sum(
            torch.nn.functional.one_hot(preds, num_classes=40) * \
            torch.nn.functional.one_hot(labels, num_classes=40), dim=0)
        mca1_den = torch.sum(
            torch.nn.functional.one_hot(labels, num_classes=40), dim=0)
        running_mca1_num.append(mca1_num.detach().cpu().numpy())
        running_mca1_den.append(mca1_den.detach().cpu().numpy())


    mca1_num = np.sum(running_mca1_num, axis=0)
    mca1_den = 1e-16 + np.sum(running_mca1_den, axis=0)

    stats = {
        'loss': np.mean(running_loss),
        'oa1': np.mean(running_oa1),
        'mca1': np.mean(mca1_num/mca1_den),
        }

    return stats

if __name__ == '__main__':
    main()
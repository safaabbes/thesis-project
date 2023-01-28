'''
This is the Source-Only Multitask Configuration where we use a source and its correct super-classes and its wrong super-classes Jointly
In this setting we apply the entropy loss on the wrong super-classes
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
import wandb

from datasets import dataset_3 as dataset
from models import resnet50_1head
from losses import loss_ce
sys.path.append('..')
from utils import get_logger

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--config', type=str, default='config3b')
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
    parser.add_argument('--mu1', type=float, default=0.33)
    parser.add_argument('--mu2', type=float, default=0.33)
    parser.add_argument('--mu3', type=float, default=0.33)

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

    # Create Wandb logger
    wandb.init(
      project='Configuration_3B', 
      name=args.exp,
      config = {"source_train": args.source_train,
                "source_test": args.source_test,
                "epochs": args.num_epochs,
                "batch_size": args.bs,
                "balance": args.balance_mini_batches,
                "lr": args.lr,
                })

    # Loop over epochs
    start = time.time()
    for epoch in range(1, args.num_epochs + 1):
        
        # Training
        since = time.time()
        stats_train = do_epoch_train(loader_train_source, model, criterion1, optimizer, args, logger)
        logger.info('TRN, Epoch: {:4d}, Loss: {:e}, OA1: {:.4f}, MCA1: {:.4f}, OA2: {:.4f}, MCA2: {:.4f}, OA3: {:.4f}, MCA3: {:.4f}, Elapsed: {:.1f}s'.format(
            epoch, stats_train['loss'], stats_train['oa1'], stats_train['mca1'], stats_train['oa2'], stats_train['mca2'], stats_train['oa3'], stats_train['mca3'], time.time() - since))
        
        # Validation
        since = time.time()
        stats_valid = do_epoch_valid(loader_valid_source, model, criterion1, args)
        logger.info('VAL, Epoch: {:4d}, Loss: {:e}, OA1: {:.4f}, MCA1: {:.4f}, Elapsed: {:.1f}s'.format(
            epoch, stats_valid['loss'], stats_valid['oa1'], stats_valid['mca1'], time.time() - since))

        # Update wandb
        update_wandb(epoch, optimizer, stats_train, stats_valid)

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
    wandb.finish()
    logger.info('Elapsed time: {:.2f} minutes'.format((end - start)/60))


def do_epoch_train(loader_train_source, model, criterion1, optimizer, args, logger):

    # Set the model in training mode
    model = model.train()

    # Init stats
    running_loss, running_source_loss1, running_source_loss2, running_source_loss3 = list(), list(), list(), list()
    running_oa1, running_mca1_num, running_mca1_den = list(), list(), list()
    running_oa2, running_mca2_num, running_mca2_den = list(), list(), list()
    running_oa3, running_mca3_num, running_mca3_den = list(), list(), list()

    for i, data in enumerate(loader_train_source):

        # Load source mini-batch
        images, labels, correct_sc, wrong_sc = data
        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        correct_sc = correct_sc.to(args.device, non_blocking=True)
        wrong_sc = wrong_sc.to(args.device, non_blocking=True)

        # Zero the parameters gradients
        optimizer.zero_grad()

        # Forward pass for source data
        logits1 = model(images)
        _, preds1 = torch.max(logits1, dim=1)
        
        # Correct Super-Class logits
        tmp1 = np.load('correct_mapping.npz', allow_pickle=True)
        correct_mapping = torch.tensor(tmp1['data'], dtype=torch.float32, device=args.device, requires_grad=False)
        logits2 = torch.mm(logits1, correct_mapping) / (1e-6 + torch.sum(correct_mapping, dim=0))
        _, preds2 = torch.max(logits2, dim=1)
        
        # Wrong Super-Class logits
        tmp2 = np.load('wrong_mapping.npz', allow_pickle=True)
        wrong_mapping = torch.tensor(tmp2['data'], dtype=torch.float32, device=args.device, requires_grad=False)
        logits3 = torch.mm(logits1, wrong_mapping) / (1e-6 + torch.sum(wrong_mapping, dim=0))
        _, preds3 = torch.max(logits3, dim=1)
        
        # Third Loss Possibilities
        # Option 1: Entropy Loss, Criterion1 is CE
        # source_loss3 = criterion1(logits3, logits3) #Explodes > Goes to NAN
        # Option 2: ||L*LT - I|| > Explodes > Goes to inf
        source_loss3 = torch.norm(torch.mm(logits3, logits3.transpose(1, 0)) - torch.eye(args.bs, device=args.device, requires_grad=False))
        # Option 3; Entropy Loss**2
        # source_loss3 = torch.pow(criterion1(logits3, logits3),2) > Goes to NAN
        
        # logger.info('source_loss3 {}'.format(source_loss3))
        # sys.exit()
        
        
        # Losses
        source_loss1 = args.mu1 * criterion1(logits1, labels)
        source_loss2 = args.mu2 * criterion1(logits2, correct_sc)
        source_loss3 = args.mu3 * source_loss3 #Entropy Loss
        logger.info('source_loss3 {}'.format(source_loss3))
        loss = source_loss1 + source_loss2 + source_loss3

        # Back-propagation
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Update losses
        running_loss.append(loss.item())
        running_source_loss1.append(source_loss1.item())
        running_source_loss2.append(source_loss2.item())
        running_source_loss3.append(source_loss3.item())

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
        
        oa2 = torch.sum(preds2 == correct_sc.squeeze()) / len(correct_sc)
        running_oa2.append(oa2.item())
        mca2_num = torch.sum(
            torch.nn.functional.one_hot(preds2, num_classes=13) * \
            torch.nn.functional.one_hot(correct_sc, num_classes=13), dim=0)
        mca2_den = torch.sum(
            torch.nn.functional.one_hot(correct_sc, num_classes=13), dim=0)
        running_mca2_num.append(mca2_num.detach().cpu().numpy())
        running_mca2_den.append(mca2_den.detach().cpu().numpy())
        
        oa3 = torch.sum(preds3 == wrong_sc.squeeze()) / len(wrong_sc)
        running_oa3.append(oa3.item())
        mca3_num = torch.sum(
            torch.nn.functional.one_hot(preds3, num_classes=9) * \
            torch.nn.functional.one_hot(wrong_sc, num_classes=9), dim=0)
        mca3_den = torch.sum(
            torch.nn.functional.one_hot(wrong_sc, num_classes=9), dim=0)
        running_mca3_num.append(mca3_num.detach().cpu().numpy())
        running_mca3_den.append(mca3_den.detach().cpu().numpy())

    # Update MCA metric
    mca1_num = np.sum(running_mca1_num, axis=0)
    mca1_den = 1e-16 + np.sum(running_mca1_den, axis=0)
    mca2_num = np.sum(running_mca2_num, axis=0)
    mca2_den = 1e-16 + np.sum(running_mca2_den, axis=0)
    mca3_num = np.sum(running_mca3_num, axis=0)
    mca3_den = 1e-16 + np.sum(running_mca3_den, axis=0)

    stats = {
        'loss': np.mean(running_loss),
        'source_loss1': np.mean(running_source_loss1),
        'source_loss2': np.mean(running_source_loss2),
        'source_loss3': np.mean(running_source_loss3),
        'oa1': np.mean(running_oa1),
        'mca1': np.mean(mca1_num/mca1_den),
        'oa2': np.mean(running_oa2),
        'mca2': np.mean(mca2_num/mca2_den),
        'oa3': np.mean(running_oa3),
        'mca3': np.mean(mca3_num/mca3_den),
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
        images, labels, _ , _= data
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


def update_wandb(epoch, optimizer, stats_train, stats_valid):

    wandb.log({
        "epoch": epoch,
        # "train/lr backbone": optimizer.param_groups[0]['lr'],
        # "train/lr head": optimizer.param_groups[1]['lr'],
        # Train Stats
        "train/loss": stats_train['loss'].item(),
        "train/source_loss1": stats_train['source_loss1'].item(),
        "train/source_loss2": stats_train['source_loss2'].item(),
        "train/source_loss3": stats_train['source_loss3'].item(),
        "train/oa1": stats_train['oa1'].item(),
        "train/mca1": stats_train['mca1'].item(),
        "train/oa2": stats_train['oa2'].item(),
        "train/mca2": stats_train['mca2'].item(),
        "train/oa3": stats_train['oa3'].item(),
        "train/mca3": stats_train['mca3'].item(),
        # Valid Stats
        "valid/loss": stats_valid['loss'].item(),
        "valid/oa1": stats_valid['oa1'].item(),
        "valid/mca1": stats_valid['mca1'].item(),
    })


if __name__ == '__main__':
    main()
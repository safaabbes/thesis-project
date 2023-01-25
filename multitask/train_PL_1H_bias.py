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
import torch.nn.functional as F
import wandb

from datasets_biased import dataset2_biased as dataset
from datasets_biased import BiasedPseudoLabelDataset
from models import resnet50s_1head
from losses import loss_ce, loss_op
sys.path.append('..')
from utils import get_logger



def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--freq_saving', type=int, default=10)
    
    # Pseudo
    parser.add_argument('--source_exp', type=str, required=True)
    parser.add_argument('--pseudo_domain', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--condition', type=str, required=True)
    
    # Train
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--balance_mini_batches', default=False, action='store_true')

    # Data
    parser.add_argument('--source_train', type=str, required=True)
    parser.add_argument('--source_test', type=str, required=True)
    parser.add_argument('--target_test', type=str, required=True)

    # Model
    parser.add_argument('--num_categories1', type=int, default=40)
    parser.add_argument('--num_categories2', type=int, default=13)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.05)

    # Optimizer
    parser.add_argument('--optim_type', type=str, default='SGD')
    parser.add_argument('--lr_init', type=float, default=1e-02)
    parser.add_argument('--lr_final', type=float, default=1e-04)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--w_decay', type=float, default=1e-05)
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--use_warmup', default=False, action='store_true')

    # Loss
    parser.add_argument('--reduction', type=str, default='mean')
    parser.add_argument('--mu1', type=float, default=0.25)
    parser.add_argument('--mu2', type=float, default=0.25)
    parser.add_argument('--mu3', type=float, default=0.25)
    parser.add_argument('--mu4', type=float, default=0.25)
    parser.add_argument('--dropout_rate', type=float, default=0.7)

    args = parser.parse_args()
    return args


def main():

    # Parse input arguments
    args = parse_args()
    

    # Update path to weights and runs
    args.path_weights = os.path.join('..', '..','data', 'exps', 'biased_models', args.exp)
    args.path_pseudo = os.path.join('..','..','data', 'exps', 'pseudo_labels', args.source_exp)
    
    # Create experiment folder
    os.makedirs(args.path_weights, exist_ok=True)

    # Create logger
    logger = get_logger(os.path.join(args.path_weights, 'log_train.txt'))

    # # Create Wandb logger
    wandb.init(dir='../',
      project='Train_PL_1H', 
      name=args.exp,
      config = {"model_type": args.model_type,
                "source_train": args.source_train,
                "source_test": args.source_test,
                "pseudo_domain": args.pseudo_domain,
                "target_test": args.target_test,
                "epochs": args.num_epochs,
                "batch_size": args.bs,
                "balance": args.balance_mini_batches,
                "lr": args.lr_init,
                "reduction": args.reduction,
                "mu1": args.mu1,
                "mu2": args.mu2,
                "mu3": args.mu3,
                "mu4": args.mu4,
                "dropout":args.dropout_rate,
                })

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

    # Get the target dataset
    pseudo_target_tensors = torch.load(os.path.join(args.path_pseudo,'{}_{}_{}.tar'.format(args.pseudo_domain, args.checkpoint, args.condition)))
    PL_file_array = pseudo_target_tensors['images_file_array']

    dataset_train_target = BiasedPseudoLabelDataset(PL_file_array, augm_type='test') 
    dataset_valid_target = dataset(
        domain_type=args.target_test,
        augm_type='test')


    # Log stats
    logger.info('Source samples, Training: {:d}, Validation: {:d}'.format(
        len(dataset_train_source), len(dataset_valid_source)))
    logger.info('Target samples, Training (Pseudo-Labels): {:d}, Validation: {:d}'.format(
        len(dataset_train_target), len(dataset_valid_target)))

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

    # Get the target dataloaders
    loader_train_target = torch.utils.data.DataLoader(
        dataset=dataset_train_target,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    loader_valid_target = torch.utils.data.DataLoader(
        dataset=dataset_valid_target,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    # Get the model
    if args.model_type.lower() == 'resnet50s_1head':
        model = resnet50s_1head(args)
    else:
        raise NotImplementedError

    # Send the model to the device
    # checkpoint = torch.load(os.path.join('..','..','data', 'exps', 'weights', args.source_exp, '{}.tar'.format(args.checkpoint)))
    # model.load_state_dict(checkpoint['model_state_dict']) 
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
                {'params': [p for n, p in params_back], 'lr': 0.1 * args.lr_init},
                {'params': [p for n, p in params_head], 'lr': args.lr_init}
            ],
            lr=args.lr_init,
            momentum=args.momentum,
            weight_decay=args.w_decay,
            nesterov=True)
    elif args.optim_type == 'Adam':
        optimizer = torch.optim.Adam(
            [
                {'params': [p for n, p in params_back], 'lr': 0.1 * args.lr_init},
                {'params': [p for n, p in params_head], 'lr': args.lr_init}
            ],
            lr=args.lr_init,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.w_decay)
    elif args.optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            [
                {'params': [p for n, p in params_back], 'lr': 0.1 * args.lr_init},
                {'params': [p for n, p in params_head], 'lr': args.lr_init}
            ],
            lr=args.lr_init,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.w_decay)
    else:
        raise NotImplementedError

    # Get the learning rate scheduler
    if args.step is None:
        args.step = args.num_epochs * len(loader_train_source)  # TODO
    if args.scheduler_type == 'step':
        scheduler_lr = torch.optim.lr_scheduler.StepLR(
            optimizer,
            args.step,  # TODO args.step - 1
            args.gamma)
    elif args.scheduler_type == 'multistep':
        scheduler_lr = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.5 * args.step), int(0.75 * args.step)],
            gamma=0.1)
    elif args.scheduler_type == 'cosine':
        scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.step,  # TODO args.step - 1
            eta_min=args.lr_final)
    else:
        raise NotImplementedError

    # Get the warmup scheduler (https://github.com/Tony-Y/pytorch_warmup)
    if args.use_warmup:
        scheduler_warmup = warmup.LinearWarmup(
            optimizer,
            warmup_period=5 * len(loader_train_source))  # TODO
    else:
        scheduler_warmup = None

    # Get losses and send them to the device
    criterion1 = loss_ce(reduction=args.reduction)
    criterion1 = criterion1.to(args.device)

    # Loop over epochs
    start = time.time()
    for epoch in range(1, args.num_epochs + 1):

        # Training
        since = time.time()
        stats_train = do_epoch_train(loader_train_source, loader_train_target, model, criterion1, optimizer, scheduler_lr, scheduler_warmup, args, logger)
        logger.info('TRN, Epoch: {:4d}, Loss: {:e}, OA1: {:.4f}, MCA1: {:.4f}, OA2: {:.4f}, MCA2: {:.4f}, Elapsed: {:.1f}s'.format(
            epoch, stats_train['loss'], stats_train['oa1'], stats_train['mca1'], stats_train['oa2'], stats_train['mca2'], time.time() - since))

        # Validation
        since = time.time()
        stats_valid = do_epoch_valid(loader_valid_source, loader_valid_target, model, criterion1, args)
        logger.info('VAL, Epoch: {:4d}, Loss: {:e}, OA1: {:.4f}, MCA1: {:.4f}, OA2: {:.4f}, MCA2: {:.4f}, Elapsed: {:.1f}s'.format(
            epoch, stats_valid['loss'], stats_valid['oa1'], stats_valid['mca1'], stats_valid['oa2'], stats_valid['mca2'], time.time() - since))

        # Update Wandb logger
        update_wandb(epoch, optimizer, stats_train, stats_valid)

        # Scheduler step
        # TODO scheduler_lr.step()

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


def do_epoch_train(loader_train_source, loader_train_target, model, criterion1, optimizer, scheduler_lr, scheduler_warmup, args, logger):

    # Set the model in training mode
    model = model.train()

    # Init stats
    running_loss, running_loss1_source, running_loss2_source, running_loss1_target, running_loss2_target = list(), list(), list(), list(), list()
    running_oa1, running_mca1_num, running_mca1_den = list(), list(), list()
    running_oa2, running_mca2_num, running_mca2_den = list(), list(), list()

    # Loop on source dataloader
    dataloader_iterator = iter(loader_train_target)
    for i, data_source in enumerate(loader_train_source):
        try:
            data_target = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(loader_train_target)
            data_target = next(dataloader_iterator)

        # Load source mini-batch
        images_source, categories1_source, categories2_source = data_source
        images_source = images_source.to(args.device, non_blocking=True)
        categories1_source = categories1_source.to(args.device, non_blocking=True)
        categories2_source = categories2_source.to(args.device, non_blocking=True)

        # Load target mini-batch
        images_target, categories1_target, categories2_target = data_target
        images_target = images_target.to(args.device, non_blocking=True)
        categories1_target = categories1_target.to(args.device, non_blocking=True)
        categories2_target = categories2_target.to(args.device, non_blocking=True)

        # Zero the parameters gradients
        optimizer.zero_grad()

        # Forward pass for source data
        logits1_source = model(images_source)
        _, preds1_source = torch.max(logits1_source, dim=1)

        tmp = np.load('mapping.npz')
        mapping = torch.tensor(tmp['data'], dtype=torch.float32, device=args.device, requires_grad=False)
        logits2_source = torch.mm(logits1_source, mapping) / (1e-6 + torch.sum(mapping, dim=0))
        _, preds2_source = torch.max(logits2_source, dim=1)

        # Forward pass for target data 
        logits1_target = model(images_target)
        _, preds1_target = torch.max(logits1_target, dim=1)
        
        tmp = np.load('mapping.npz')
        mapping = torch.tensor(tmp['data'], dtype=torch.float32, device=args.device, requires_grad=False)
        logits2_target = torch.mm(logits1_target, mapping) / (1e-6 + torch.sum(mapping, dim=0))
        _, preds2_target = torch.max(logits2_target, dim=1)
        
        # Dropout Some Pseudo Labels
        mask = torch.rand(logits1_target.shape[0]) < args.dropout_rate
        logits1_dropped = logits1_target[mask]
        logits2_dropped = logits2_target[mask]
        targets1_dropped = categories1_target[mask]
        targets2_dropped = categories2_target[mask]

        # Losses
        loss1_source = args.mu1 * criterion1(logits1_source, categories1_source)
        loss2_source = args.mu2 * criterion1(logits2_source, categories2_source)
        loss1_target = args.mu3 * criterion1(logits1_dropped, targets1_dropped)
        loss2_target = args.mu4 * criterion1(logits2_dropped, targets2_dropped)
        loss = loss1_source + loss2_source + loss1_target + loss2_target
        # logger.info('loss {}'.format(loss))

        # Back-propagation
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Scheduler step
        if args.use_warmup:
            with scheduler_warmup.dampening():
                scheduler_lr.step()
        else:
            scheduler_lr.step()

        # Update losses
        running_loss.append(loss.item())
        running_loss1_source.append(loss1_source.item())
        running_loss2_source.append(loss2_source.item())
        running_loss1_target.append(loss1_target.item())
        running_loss2_target.append(loss2_target.item())

        # Update metrics
        oa1 = torch.sum(preds1_source == categories1_source.squeeze()) / len(categories1_source)
        running_oa1.append(oa1.item())
        mca1_num = torch.sum(
            torch.nn.functional.one_hot(preds1_source, num_classes=args.num_categories1) * \
            torch.nn.functional.one_hot(categories1_source, num_classes=args.num_categories1), dim=0)
        mca1_den = torch.sum(
            torch.nn.functional.one_hot(categories1_source, num_classes=args.num_categories1), dim=0)
        running_mca1_num.append(mca1_num.detach().cpu().numpy())
        running_mca1_den.append(mca1_den.detach().cpu().numpy())

        oa2 = torch.sum(preds2_source == categories2_source.squeeze()) / len(categories2_source)
        running_oa2.append(oa2.item())
        mca2_num = torch.sum(
            torch.nn.functional.one_hot(preds2_source, num_classes=args.num_categories2) * \
            torch.nn.functional.one_hot(categories2_source, num_classes=args.num_categories2), dim=0)
        mca2_den = torch.sum(
            torch.nn.functional.one_hot(categories2_source, num_classes=args.num_categories2), dim=0)
        running_mca2_num.append(mca2_num.detach().cpu().numpy())
        running_mca2_den.append(mca2_den.detach().cpu().numpy())

    # Update MCA metric
    mca1_num = np.sum(running_mca1_num, axis=0)
    mca1_den = 1e-16 + np.sum(running_mca1_den, axis=0)
    mca2_num = np.sum(running_mca2_num, axis=0)
    mca2_den = 1e-16 + np.sum(running_mca2_den, axis=0)

    stats = {
        'loss': np.mean(running_loss),
        'loss1_source': np.mean(running_loss1_source),
        'loss2_source': np.mean(running_loss2_source),
        'loss1_target': np.mean(running_loss1_target),
        'loss2_target': np.mean(running_loss2_target),
        'oa1': np.mean(running_oa1),
        'mca1': np.mean(mca1_num/mca1_den),
        'oa2': np.mean(running_oa2),
        'mca2': np.mean(mca2_num/mca2_den)
        }

    return stats


def do_epoch_valid(loader_valid_source, loader_valid_target, model, criterion1, args):

    # Set the model in evaluation mode
    model = model.eval()

    # Init stats
    running_loss, running_loss1_source, running_loss2_source, running_loss1_target, running_loss2_target = list(), list(), list(), list(), list()
    running_oa1, running_mca1_num, running_mca1_den = list(), list(), list()
    running_oa2, running_mca2_num, running_mca2_den = list(), list(), list()

    # Loop over validation mini-batches
    for data_source, data_target in zip(loader_valid_source, loader_valid_target):

        # Load source mini-batch
        images_source, categories1_source, categories2_source = data_source
        images_source = images_source.to(args.device, non_blocking=True)
        categories1_source = categories1_source.to(args.device, non_blocking=True)
        categories2_source = categories2_source.to(args.device, non_blocking=True)

        # Load target mini-batch
        images_target, categories1_target, categories2_target = data_target
        images_target = images_target.to(args.device, non_blocking=True)
        categories1_target = categories1_target.to(args.device, non_blocking=True)
        categories2_target = categories2_target.to(args.device, non_blocking=True)

        with torch.inference_mode():

            # Forward pass for source data
            logits1_source = model(images_source)
            _, preds1_source = torch.max(logits1_source, dim=1)

            tmp = np.load('mapping.npz')
            mapping = torch.tensor(tmp['data'], dtype=torch.float32, device=args.device, requires_grad=False)
            logits2_source = torch.mm(logits1_source, mapping) / (1e-6 + torch.sum(mapping, dim=0))
            _, preds2_source = torch.max(logits2_source, dim=1)

            # Forward pass for target data
            logits1_target = model(images_target)
            _, preds1_target = torch.max(logits1_target, dim=1)

            tmp = np.load('mapping.npz')
            mapping = torch.tensor(tmp['data'], dtype=torch.float32, device=args.device, requires_grad=False)
            logits2_target = torch.mm(logits1_target, mapping) / (1e-6 + torch.sum(mapping, dim=0))
            _, preds2_target = torch.max(logits2_target, dim=1)

            # Dropout Some Pseudo Labels
            mask = torch.rand(logits1_target.shape[0]) < args.dropout_rate
            logits1_dropped = logits1_target[mask]
            logits2_dropped = logits2_target[mask]
            targets1_dropped = categories1_target[mask]
            targets2_dropped = categories2_target[mask]

            # Losses
            loss1_source = args.mu1 * criterion1(logits1_source, categories1_source)
            loss2_source = args.mu2 * criterion1(logits2_source, categories2_source)
            loss1_target = args.mu3 * criterion1(logits1_dropped, targets1_dropped)
            loss2_target = args.mu4 * criterion1(logits2_dropped, targets2_dropped)
            loss = loss1_source + loss2_source + loss1_target + loss2_target

        # Update losses
        running_loss.append(loss.item())
        running_loss1_source.append(loss1_source.item())
        running_loss2_source.append(loss2_source.item())
        running_loss1_target.append(loss1_target.item())
        running_loss2_target.append(loss2_target.item())

        # Update metrics
        oa1 = torch.sum(preds1_source == categories1_source.squeeze()) / len(categories1_source)
        running_oa1.append(oa1.item())
        mca1_num = torch.sum(
            torch.nn.functional.one_hot(preds1_source, num_classes=args.num_categories1) * \
            torch.nn.functional.one_hot(categories1_source, num_classes=args.num_categories1), dim=0)
        mca1_den = torch.sum(
            torch.nn.functional.one_hot(categories1_source, num_classes=args.num_categories1), dim=0)
        running_mca1_num.append(mca1_num.detach().cpu().numpy())
        running_mca1_den.append(mca1_den.detach().cpu().numpy())

        oa2 = torch.sum(preds2_source == categories2_source.squeeze()) / len(categories2_source)
        running_oa2.append(oa2.item())
        mca2_num = torch.sum(
            torch.nn.functional.one_hot(preds2_source, num_classes=args.num_categories2) * \
            torch.nn.functional.one_hot(categories2_source, num_classes=args.num_categories2), dim=0)
        mca2_den = torch.sum(
            torch.nn.functional.one_hot(categories2_source, num_classes=args.num_categories2), dim=0)
        running_mca2_num.append(mca2_num.detach().cpu().numpy())
        running_mca2_den.append(mca2_den.detach().cpu().numpy())

    mca1_num = np.sum(running_mca1_num, axis=0)
    mca1_den = 1e-16 + np.sum(running_mca1_den, axis=0)
    mca2_num = np.sum(running_mca2_num, axis=0)
    mca2_den = 1e-16 + np.sum(running_mca2_den, axis=0)

    stats = {
        'loss': np.mean(running_loss),
        'loss1_source': np.mean(running_loss1_source),
        'loss2_source': np.mean(running_loss2_source),
        'loss1_target': np.mean(running_loss1_target),
        'loss2_target': np.mean(running_loss2_target),
        'oa1': np.mean(running_oa1),
        'mca1': np.mean(mca1_num/mca1_den),
        'oa2': np.mean(running_oa2),
        'mca2': np.mean(mca2_num/mca2_den)
        }

    return stats


def update_wandb(epoch, optimizer, stats_train, stats_valid):

    wandb.log({
        "epoch": epoch,
        "train/lr backbone": optimizer.param_groups[0]['lr'],
        "train/lr head": optimizer.param_groups[1]['lr'],
        # Train Stats
        "train/loss": stats_train['loss'].item(),
        "train/loss1_source": stats_train['loss1_source'].item(),
        "train/loss2_source": stats_train['loss2_source'].item(),
        "train/loss1_target": stats_train['loss1_target'].item(),
        "train/loss2_target": stats_train['loss2_target'].item(),
        "train/oa1": stats_train['oa1'].item(),
        "train/mca1": stats_train['mca1'].item(),
        "train/oa2": stats_train['oa2'].item(),
        "train/mca2": stats_train['mca2'].item(),
        # Valid Stats
        "valid/loss": stats_valid['loss'].item(),
        "valid/loss1_source": stats_valid['loss1_source'].item(),
        "valid/loss2_source": stats_valid['loss2_source'].item(),
        "valid/loss1_target": stats_valid['loss1_target'].item(),
        "valid/loss2_target": stats_valid['loss2_target'].item(),
        "valid/oa1": stats_valid['oa1'].item(),
        "valid/mca1": stats_valid['mca1'].item(),
        "valid/oa2": stats_valid['oa2'].item(),
        "valid/mca2": stats_valid['mca2'].item(),
    })


if __name__ == '__main__':
    main()
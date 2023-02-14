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
import wandb

import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

from datasets import dataset
from models import resnet50_1h
from losses import loss_ce
sys.path.append('..')
from utils import get_logger

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', nargs='+', type=int)
    parser.add_argument('--freq_saving', type=int, default=10)

    # Train
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--balance_mini_batches', default=False, action='store_true')

    parser.add_argument('--prev_arch', type=str, required=True)
    parser.add_argument('--prev_exp', type=str, required=True)
    parser.add_argument('--prev_seed', type=str, required=True)
    parser.add_argument('--prev_checkpoint', type=str, default='0040')

    # Data
    parser.add_argument('--source_train', type=str, required=True)
    parser.add_argument('--source_test', type=str, required=True)

    # Model
    parser.add_argument('--num_categories1', type=int, default=40)
    parser.add_argument('--num_categories2', type=int, default=13)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.05)

    # Optimizer
    parser.add_argument('--optim_type', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=1e-02)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--w_decay', type=float, default=1e-05)
    
    
    parser.add_argument('--mu1', type=float, default=0.5)
    parser.add_argument('--mu2', type=float, default=0.5)

    args = parser.parse_args()
    return args


def main():

    # Parse input arguments
    args = parse_args()

    # Update path to weights and runs
    args.path_weights = os.path.join('..', '..','data', 'new_exps', 'configa', args.exp)
    args.prev_exp = os.path.join('..','..', 'data', 'new_exps', args.prev_arch, args.prev_exp, args.prev_seed)
    
    # Create experiment folder
    os.makedirs(args.path_weights, exist_ok=True)

    # Create logger
    logger = get_logger(os.path.join(args.path_weights, 'log_train.txt'))

    # Log library versions
    logger.info('PyTorch version = {:s}'.format(torch.__version__))
    logger.info('TorchVision version = {:s}'.format(torchvision.__version__))

    # Activate CUDNN backend
    torch.backends.cudnn.enabled = True
    
    # Log input arguments
    for arg, value in vars(args).items():
        logger.info('{:s} = {:s}'.format(arg, str(value)))

    for seed in args.seed:
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        logger.info('Train with fixed seed: {}'.format(seed))
        # Perform the training
        run_train(args, logger, seed)


def run_train(args, logger, seed):
    
    args.path_weights = os.path.join('..', '..','data', 'new_exps', 'configa', args.exp , str(seed))
    os.makedirs(args.path_weights, exist_ok=True)

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
    if args.model_type.lower() == 'resnet50_1h':
        model = resnet50_1h(args)
    else:
        raise NotImplementedError

    prev_checkpoint = torch.load(os.path.join(args.prev_exp, '{:s}.tar'.format(args.prev_checkpoint)))
    model.load_state_dict(prev_checkpoint['model_state_dict'])
    
    # Send the model to the device
    model = model.to(args.device)
    # logger.info('Model is on device: {}'.format(next(model.parameters()).device))

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
    
    # # Create Wandb logger
    wandb.init(dir='../',
      project='ConfigA', 
      name='{}_{}'.format(args.exp, seed),
      config = {"model_type": args.model_type,
                "source_train": args.source_train,
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
        stats_train = do_epoch_train(loader_train_source, model, criterion1, optimizer, args)
        logger.info('TRN, Epoch: {:4d}, Loss: {:e}, OA1: {:.4f}, MCA1: {:.4f}, Elapsed: {:.1f}s'.format(
            epoch, stats_train['loss'], stats_train['oa1'], stats_train['mca1'], time.time() - since))

        # Validation
        since = time.time()
        stats_valid = do_epoch_valid(loader_valid_source, model, criterion1, args)
        logger.info('VAL, Epoch: {:4d}, Loss: {:e}, OA1: {:.4f}, MCA1: {:.4f}, Elapsed: {:.1f}s'.format(
            epoch, stats_valid['loss'], stats_valid['oa1'], stats_valid['mca1'], time.time() - since))

        # Update Wandb logger
        update_wandb(epoch, stats_train, stats_valid)
        get_distances(epoch, model, args, wandb)

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


def do_epoch_train(loader_train_source, model, criterion1, optimizer, args):

    # Set the model in training mode
    model = model.train()

    # Init stats
    running_loss, running_source_loss1, running_source_loss2 = list(), list(), list()
    running_oa1, running_mca1_num, running_mca1_den = list(), list(), list()
    running_oa2, running_mca2_num, running_mca2_den = list(), list(), list()
    # Loop on source dataloader. Source: https://stackoverflow.com/questions/51444059/how-to-iterate-over-two-dataloaders-simultaneously-using-pytorch
    for i, data_source in enumerate(loader_train_source):

        # Load source mini-batch
        images_source, categories1_source, categories2_source = data_source
        images_source = images_source.to(args.device, non_blocking=True)
        categories1_source = categories1_source.to(args.device, non_blocking=True)
        categories2_source = categories2_source.to(args.device, non_blocking=True)

        # Zero the parameters gradients
        optimizer.zero_grad()

        # Forward pass for source data
        logits1_source = model(images_source)
        _, preds1_source = torch.max(logits1_source, dim=1)
        
        tmp = np.load('mapping.npz')
        mapping = torch.tensor(tmp['data'], dtype=torch.float32, device=args.device, requires_grad=False)
        logits2_source = torch.mm(logits1_source, mapping) / (1e-6 + torch.sum(mapping, dim=0))
        _, preds2_source = torch.max(logits2_source, dim=1)
        
        # Losses
        source_loss1 = args.mu1 * criterion1(logits1_source, categories1_source)  
        source_loss2 = args.mu2 * criterion1(logits2_source, categories2_source)  
        loss = source_loss1 + source_loss2
        
        # Back-propagation
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Update losses
        running_loss.append(loss.item())
        running_source_loss1.append(source_loss1.item())
        running_source_loss2.append(source_loss2.item())

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
        'source_loss1': np.mean(running_source_loss1),
        'source_loss2': np.mean(running_source_loss2),
        'oa1': np.mean(running_oa1),
        'mca1': np.mean(mca1_num/mca1_den),
        'oa2': np.mean(running_oa2),
        'mca2': np.mean(mca2_num/mca2_den),
        }

    return stats


def do_epoch_valid(loader_valid_source, model, criterion1, args):

    # Set the model in evaluation mode
    model = model.eval()

    # Init stats
    running_loss = list()
    running_oa1, running_mca1_num, running_mca1_den = list(), list(), list()

    # Loop over validation mini-batches
    for data_source in loader_valid_source:

        # Load source mini-batch
        images_source, categories1_source, _ = data_source
        images_source = images_source.to(args.device, non_blocking=True)
        categories1_source = categories1_source.to(args.device, non_blocking=True)

        with torch.inference_mode():

            # Forward pass for source data
            logits1_source = model(images_source)
            _, preds1_source = torch.max(logits1_source, dim=1)

            # Losses
            loss = criterion1(logits1_source, categories1_source) 

        # Update losses
        running_loss.append(loss.item())

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

    mca1_num = np.sum(running_mca1_num, axis=0)
    mca1_den = 1e-16 + np.sum(running_mca1_den, axis=0)

    stats = {
        'loss': np.mean(running_loss),
        'oa1': np.mean(running_oa1),
        'mca1': np.mean(mca1_num/mca1_den),
        }

    return stats


def update_wandb(epoch, stats_train, stats_valid):

    wandb.log({
        "epoch": epoch,
        # Train Stats
        "train/loss": stats_train['loss'].item(),
        "train/source_loss1": stats_train['source_loss1'].item(),
        "train/source_loss2": stats_train['source_loss2'].item(),
        "train/oa1": stats_train['oa1'].item(),
        "train/mca1": stats_train['mca1'].item(),
        "train/oa2": stats_train['oa2'].item(),
        "train/mca2": stats_train['mca2'].item(),
        # Valid Stats
        "valid/loss": stats_valid['loss'].item(),
        "valid/oa1": stats_valid['oa1'].item(),
        "valid/mca1": stats_valid['mca1'].item(),
    })

def get_distances(epoch, model, args, wandb):
    
    # Get head weights (40,2080)
    head_weights = model.head.weight.data                   #Shape (40,2080)
    transposed_head_weights = torch.transpose(head_weights, 0, 1)      #Shape (2080,40)
    # Get mapping from C1 to C2 (40,13)
    tmp = np.load('mapping.npz')
    mapping = torch.tensor(tmp['data'], dtype=torch.float32, device= args.device, requires_grad=False)   #Shape (40,13)
    # Get SC weights (2080,40)
    sc_weights = torch.mm(transposed_head_weights, mapping) / (1e-6 + torch.sum(mapping, dim=0))   #Shape (2080,13)
    
    # Get Average of all pairwise distances
    pair_dist = F.pdist(head_weights)
    distance_all = torch.mean(pair_dist)
    
    # Get Average of inter-cluster distances
    sc_weights = torch.transpose(sc_weights, 0, 1)   #Shape (13,2080)
    sc_pair_dist = F.pdist(sc_weights)
    distance_inter_cluster = torch.mean(sc_pair_dist)
    
    # Get Intra-Cluster of each cluster (distance relative to the centroid)
    mask = torch.tensor(tmp['data'], dtype=torch.bool, requires_grad=False) 
    for cluster in range(args.num_categories2):
        intra_cluster_dist = torch.mean(torch.cdist(sc_weights[cluster,:].unsqueeze(0), head_weights[mask[:,cluster]]))
        wandb.log({
            "distance/epoch": epoch,
            "distance/intra_cluster_{}".format(cluster): intra_cluster_dist.item()})
    
    wandb.log({
        "distance/epoch": epoch,
        "distance/pairwise_all": distance_all.item(),
        "distance/inter_cluster": distance_inter_cluster.item(),
        })
    
    # Plot distance matrix
    distance_mat = pairwise_distances(head_weights.cpu().numpy())
    sc_distance_mat = pairwise_distances(sc_weights.cpu().numpy())
    plot_dist_matrix(distance_mat, epoch, wandb, args)
    plot_dist_matrix(sc_distance_mat, epoch, wandb, args)
    
    # # Get Distances for 3 Sample Clusters
    # mask = torch.tensor(tmp['data'], dtype=torch.bool, requires_grad=False) 
    # cluster_0_weights = head_weights[mask[:,0]] # Furniture Cluster 5 instances
    # cluster_1_weights = head_weights[mask[:,1]] # Mammal Cluster 9 instances
    # cluster_5_weights = head_weights[mask[:,5]] # Road Transport Cluster 5 instances
    # intra_cluster_0 = torch.mean(F.pdist(cluster_0_weights))
    # intra_cluster_1 = torch.mean(F.pdist(cluster_1_weights))
    # intra_cluster_5 = torch.mean(F.pdist(cluster_5_weights))
    # # Plot Sample Cluster
    # intra_cluster_0_mat = pairwise_distances(cluster_0_weights.cpu().numpy())
    # intra_cluster_1_mat = pairwise_distances(cluster_1_weights.cpu().numpy())
    # intra_cluster_5_mat = pairwise_distances(cluster_5_weights.cpu().numpy())
    # plot_dist_matrix_sample(intra_cluster_0_mat, epoch, wandb, args, 'intra_cluster_furniture_mat')
    # plot_dist_matrix_sample(intra_cluster_1_mat, epoch, wandb, args, 'intra_cluster_mammal_mat')
    # plot_dist_matrix_sample(intra_cluster_5_mat, epoch, wandb, args, 'intra_cluster_road_transport_mat')
    # wandb.log({
    #     "distance/intra_cluster_furniture": intra_cluster_0,
    #     "distance/inter_cluster_mammal": intra_cluster_1,
    #     "distance/inter_cluster_road_transport": intra_cluster_5,
    #     })
     
# def plot_dist_matrix_sample(distance_mat, epoch, wandb, args, title):
    
#     # distance_mat = distance_mat / np.sum(distance_mat, axis=0)
    
#     plt.figure(figsize=(25, 25))
#     plt.imshow(np.around(distance_mat, decimals=2), cmap='Blues')

#     plt.title('Distance Matrix {}_{}_{}'.format(title, args.exp, epoch))
#     plt.xlabel('Classes')
#     plt.ylabel('Classes')
        
#     plt.colorbar()
#     ax = plt.gca()
#     ax.set_xticks(np.arange(len(distance_mat)))
#     ax.set_yticks(np.arange(len(distance_mat)))
#     ax.set_xticklabels(np.arange(len(distance_mat)))
#     ax.set_yticklabels(np.arange(len(distance_mat)))
    
#     for i in range(len(distance_mat)):
#         for j in range(len(distance_mat)):
#             text = plt.text(j, i, round(distance_mat[i, j], 2),
#                         ha="center", va="center", color="black", fontsize=30)
    
#     # plt.savefig(os.path.join(args.path_weights,'dist_{}.png'.format(epoch)), format='png', dpi=300)
#     # np.savez(os.path.join(args.path_weights, 'dist_{}.npz'.format(epoch)), data=distance_mat)
#     wandb.log({"distance/{}".format(title): wandb.Image(plt)})
#     plt.close()    
    

def plot_dist_matrix(distance_mat, epoch, wandb, args):
    
    # distance_mat = distance_mat / np.sum(distance_mat, axis=0)
    if len(distance_mat) == args.num_categories1:
        plt.figure(figsize=(40, 40))
    elif len(distance_mat) == args.num_categories2:
        plt.figure(figsize=(20, 20))
    plt.imshow(np.around(distance_mat, decimals=2), cmap='Blues')

    if len(distance_mat) == args.num_categories1:
        plt.title('Distance Matrix {}_{}'.format(args.exp, epoch))
        plt.xlabel('Classes')
        plt.ylabel('Classes')
    elif len(distance_mat) == args.num_categories2:
        plt.title('Inter-Cluster Distance Matrix {}_{}'.format(args.exp, epoch))
        plt.xlabel('Super-Classes')
        plt.ylabel('Super-Classes')
        
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(np.arange(len(distance_mat)))
    ax.set_yticks(np.arange(len(distance_mat)))
    ax.set_xticklabels(np.arange(len(distance_mat)))
    ax.set_yticklabels(np.arange(len(distance_mat)))
    
    for i in range(len(distance_mat)):
        for j in range(len(distance_mat)):
            text = plt.text(j, i, round(distance_mat[i, j], 2),
                        ha="center", va="center", color="black", fontsize=15)
    
    if len(distance_mat) == args.num_categories1:
        # plt.savefig(os.path.join(args.path_weights,'dist_{}.png'.format(epoch)), format='png', dpi=300)
        # np.savez(os.path.join(args.path_weights, 'dist_{}.npz'.format(epoch)), data=distance_mat)
        wandb.log({"distance/dist": wandb.Image(plt)})
    elif len(distance_mat) == args.num_categories2:
        # plt.savefig(os.path.join(args.path_weights,'sc_dist_{}.png'.format(epoch)), format='png', dpi=300)
        # np.savez(os.path.join(args.path_weights, 'sc_dist_{}.npz'.format(epoch)), data=distance_mat)
        wandb.log({"distance/sc_dist": wandb.Image(plt)})
        
    plt.close()

if __name__ == '__main__':
    main()
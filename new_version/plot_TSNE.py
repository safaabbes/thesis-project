import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
from pylab import cm
from matplotlib import colors
import sys

import torch
import torchvision
from sklearn.manifold import TSNE

from datasets import dataset
from models import resnet50_1h, resnet50_2h
sys.path.append('..')
from utils import get_logger

from create_splits import CATEGORY_NAMES, SC_CATEGORY_NAMES

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--train_seed', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
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
    args_test.path_weights = os.path.join('..','..', 'data', 'new_exps',  args_test.arch, args_test.exp, args_test.train_seed)

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
    path_log = os.path.join(args.path_weights, 'tsne_{:s}_{:s}.txt'.format(args.test_domain, args.checkpoint))
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
    run_tsne(args, logger, checkpoint)


def run_tsne(args, logger, checkpoint):

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
    if args.model_type.lower() == 'resnet50_1h':
        model = resnet50_1h(args)
    elif args.model_type.lower() == 'resnet50_2h':
        model = resnet50_2h(args)
    else:
        raise NotImplementedError

    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    feature_extractor = model.backbone
    
    # Send model to device
    feature_extractor.to(args.device)

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

    class_to_sc = np.unique(labels_list, axis=0)
    class_labels = labels_list[:,0]
    sc_labels = labels_list[:,1]
    cmap = cm.get_cmap('tab20', 13)
    markers = ["${}$".format(i) for i in range(40)]
    
    # 2D embedding using TSNE
    tsne = TSNE(2, verbose=1, perplexity= 40, random_state=1234)
    features_viz = tsne.fit_transform(features)
    fig = plt.figure()
    ax = fig.add_subplot()
    for c in np.unique(class_labels):                
        _features = features_viz[c == class_labels]
        ax.scatter(_features[:, 0], _features[:, 1], marker=markers[c],
                    s=50, alpha=0.75, c=np.array([cmap(class_to_sc[c,1] % 13)]))
        
    leg = ax.legend(CATEGORY_NAMES, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.60))
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.savefig(os.path.join(args.path_weigths, fn), transparent=False, bbox_inches='tight', dpi=300)
    plt.close()
                            
if __name__ == '__main__':
    main()

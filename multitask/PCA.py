import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
import sys

import torch
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

import torchvision.models as models
from datasets import dataset2 as dataset
from models import resnet50s
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

    # Model
    parser.add_argument('--checkpoint', type=str, required=True)

    args = parser.parse_args()
    return args


def main():

    # Parse input arguments
    args_test = parse_args()

    # Update path to weights and runs
    args_test.path_weights = os.path.join('..','..', 'data', 'exps', 'models', args_test.exp)

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
    path_log = os.path.join(args.path_weights, 'log_test_{:s}_{:s}.txt'.format(args.test_domain, args.checkpoint))
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
        domain_type='{:s}_test'.format(args.test_domain),
        augm_type='test')
    logger.info('Test samples: {:d}'.format(len(dataset_test)))
    
    # Get the test dataloader
    loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False)

    # Get model
    if args.model_type.lower() == 'resnet50s':
        model = resnet50s(args)
    else:
        raise NotImplementedError

    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # extract the weights of the classifier layer
    head1 = model.head1
    head2 = model.head2
    
    # convert the weights tensor to a numpy array
    weights_head1 = head1.weight.data.numpy()
    weights_head2 = head2.weight.data.numpy()

    # create an instance of PCA
    pca = PCA(n_components=3)

    # fit the PCA on the weights
    pca.fit(weights_head1)
    pca.fit(weights_head2)

    # transform the weights
    weights_head1_pca = pca.transform(weights_head1)
    weights_head2_pca = pca.transform(weights_head2)
    
    # create a scatter plot for the first set of weights
    trace1 = go.Scatter3d(x=weights_head1_pca[:, 0], y=weights_head1_pca[:, 1], z=weights_head1_pca[:, 2],
                        mode='markers',
                        marker=dict(size=5,
                                    color='green',
                                    symbol='circle',
                                    line=dict(color='rgb(50,50,50)', width=0.5)))

    # create a scatter plot for the second set of weights
    trace2 = go.Scatter3d(x=weights_head2_pca[:, 0], y=weights_head2_pca[:, 1], z=weights_head2_pca[:, 2],
                        mode='markers',
                        marker=dict(size=5,
                                    color='blue',
                                    symbol='circle',
                                    line=dict(color='rgb(50,50,50)', width=0.5)))

    fig = go.Figure(data=[trace1,trace2])
    
    pio.write_html(fig, file='pca_plot_3d.html', auto_open=False)
    
    # # create an instance of PCA
    # pca = PCA(n_components=2)

    # # fit the PCA on the weights
    # pca.fit(weights)

    # transform the weights
    # weights_pca = pca.transform(weights)

    # # Send model to device
    # model.to(args.device)
    # logger.info('Model is on device: {}'.format(next(model.parameters()).device))

    # # Put model in evaluation mode
    # model.eval()

    # # create a new model that only includes the desired layer
    # layer4 = nn.Sequential(*list(model.children())[:9])

    # logger.info('layer 4 {}'.format(layer4))
    # # create an instance of PCA
    # pca = decompose.PCA(n_components=2)

    # # test the model
    # for i, data in enumerate(test_loader):
    #     inputs, labels = data
    #     features = layer4(inputs)
    #     features = features.view(features.size(0), -1)
    #     features = pca(features)
    #     # visualize features
    #     visualize_pca(features)
    




    # # visualize the weights using PCA
    # visualize_pca(weights_pca)

if __name__ == '__main__':
    main()



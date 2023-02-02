import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances

from models import resnet50_1h


def parse_args():
    parser = argparse.ArgumentParser()
    
    # General
    parser.add_argument('--exp', type=str, default= 'Sketch_00')
    parser.add_argument('--arch', type=str, default= 'baseline')
    parser.add_argument('--seed', type=str, default= '1234')
    parser.add_argument('--checkpoint', type=str, default= '0040')
    

    args = parser.parse_args()
    return args

def main():
    
    args_test = parse_args()
    args_test.path_weights = os.path.join('..', '..','data', 'new_exps',  args_test.arch, args_test.exp, args_test.seed)
    
    checkpoint = torch.load(os.path.join(args_test.path_weights, '{:s}.tar'.format(args_test.checkpoint)))
    args = checkpoint['args']
    
    # Update training arguments
    args.exp = args_test.exp
    args.checkpoint = args_test.checkpoint
    args.path_weights = args_test.path_weights
    args.seed = args_test.seed
    args.arch = args_test.arch

    # Get model
    model = resnet50_1h(args)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get head weights (40,2080)
    head_weights = model.head.weight.data                   #Shape (40,2080)
    transposed_head_weights = torch.transpose(head_weights, 0, 1)      #Shape (2080,40)
    
    # Get mapping from C1 to C2 (40,13)
    tmp = np.load('mapping.npz')
    mapping = torch.tensor(tmp['data'], dtype=torch.float32, requires_grad=False) 
    sc_weights = torch.mm(transposed_head_weights, mapping) / (1e-6 + torch.sum(mapping, dim=0))   #Shape (2080,13)
    sc_weights = torch.transpose(sc_weights, 0, 1)   #Shape (13,2080)
    
    mask = torch.tensor(tmp['data'], dtype=torch.bool, requires_grad=False)   #Shape (40,13)
    # Get SC weights (2080,40)
    cluster_0_weights = head_weights[mask[:,0]] # Furniture Cluster 5 instances
    cluster_1_weights = head_weights[mask[:,1]] # Mammal Cluster 9 instances
    cluster_5_weights = head_weights[mask[:,5]]
    cluster_7_weights = head_weights[mask[:,7]]
    # Road Transport Cluster 5 instances
    
    
    print(sc_weights.shape)
    print(sc_weights[7,:].shape)
    test = torch.cdist(sc_weights[7,:].unsqueeze(0), cluster_7_weights)
    print(test.shape)
    mean_test = torch.mean(test)
    print(mean_test)
    
    
    
    
    # print(cluster_0_weights.shape)
    # print(cluster_1_weights.shape)
    # print(cluster_5_weights.shape)
    
    
    
    # intra_cluster_0 = torch.mean(F.pdist(cluster_0_weights))
    # intra_cluster_1 = torch.mean(F.pdist(cluster_1_weights))
    # intra_cluster_5 = torch.mean(F.pdist(cluster_5_weights))
    # print(intra_cluster_0)
    # print(intra_cluster_1)
    # print(intra_cluster_5)
    
    # intra_cluster_0_mat = pairwise_distances(cluster_0_weights.cpu().numpy())
    # intra_cluster_1_mat = pairwise_distances(cluster_1_weights.cpu().numpy())
    # intra_cluster_5_mat = pairwise_distances(cluster_5_weights.cpu().numpy())
    # print(intra_cluster_0_mat)
    # print(intra_cluster_1_mat)
    # print(intra_cluster_5_mat)
        
    # # # 
    
    # # Get Average of all pairwise distances
    # pair_dist = F.pdist(head_weights)
    # distance_all = torch.mean(pair_dist)
    # print(pair_dist)
    # print(pair_dist.shape)
    # print(distance_all)
    
    # # Get Average of inter-cluster distances
    # sc_weights = torch.transpose(sc_weights, 0, 1)   #Shape (13,2080)
    # sc_pair_dist = F.pdist(sc_weights)
    # distance_inter_cluster = torch.mean(sc_pair_dist)
    # print(sc_pair_dist)
    # print(sc_pair_dist.shape)
    # print(distance_inter_cluster)
    
    # # Create the distance matrix
    # distance_mat = pairwise_distances(head_weights.cpu().numpy())
    # print(distance_mat.shape)
    # print(distance_mat)
    # sc_distance_mat = pairwise_distances(sc_weights.cpu().numpy())
    # print(sc_distance_mat.shape)
    # print(sc_distance_mat)



if __name__ == '__main__':
    main()


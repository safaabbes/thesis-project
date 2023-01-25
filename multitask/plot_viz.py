import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from pylab import cm
from matplotlib import colors
import torch
import sys

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from create_splits import CATEGORY_NAMES, SC_CATEGORY_NAMES


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--filenames', nargs='+', default=[], required=True)
    parser.add_argument('--nb_head', type=str, required=True)
    parser.add_argument('--viz', type=str, choices=['PCA', 'TSNE'], required=True)


    args = parser.parse_args()
    return args


def main():

    # Parse input arguments
    args = parse_args()

    # Update path
    args.path_weigths = os.path.join('..', '..', 'data', 'exps', 'models', args.exp)

    
    for args.do_norm in [False]:

        for filename in args.filenames:

            # Get learned features
            data = np.load(os.path.join(args.path_weigths, '{:s}.npz'.format(filename)))
            if args.nb_head == 1:
                class_weights = data['weights_head']
            elif args.nb_head == 2:
                class_weights = data['weights_head1']
                sc_weights = data['weights_head2']
            features = data['features']
            labels = data['labels']
        
        
            class_to_sc = np.unique(labels, axis=0)
            class_labels = labels[:,0]
            sc_labels = labels[:,1]
            

            # Get colormaps
            # cmap = sns.color_palette(cc.glasbey, n_colors=13)
            # cmap = cm.get_cmap(cc.glasbey, 40)
            cmap = cm.get_cmap('tab20', 13)
            markers = ["${}$".format(i) for i in range(40)]

            
            if args.viz == "PCA":
                # 2D embedding using PCA
                features_viz, _ = pca(features, no_dims=2)
                # sc_weights_viz, _ = pca(sc_weights, no_dims=2)
                # class_weights_viz, _ = pca(class_weights, no_dims=2)
                
                # Other approach with sklearn library (same result)
                # pca = PCA(n_components=2)
                # pca.fit(features)
                # pca.fit(sc_weights)
                # pca.fit(class_weights)
                # features_viz = sklearn_pca.transform(features)
                # sc_weights_viz = sklearn_pca.transform(sc_weights)
                # class_weights_viz = sklearn_pca.transform(class_weights)
            
            elif args.viz == "TSNE":
                # 2D embedding using TSNE
                tsne = TSNE(2, verbose=1, perplexity= 40, random_state=1234)
                features_viz = tsne.fit_transform(features)
                # sc_weights_viz = tsne.fit_transform(sc_weights)
                # class_weights_viz = tsne.fit_transform(class_weights)
            
            # Normalize features
            if args.do_norm:
                features_viz = np.divide(features_viz, np.expand_dims(np.linalg.norm(features_viz, ord=2, axis=-1), axis=-1))
                # sc_weights_viz = np.divide(sc_weights_viz, np.expand_dims(np.linalg.norm(sc_weights_viz, ord=2, axis=-1), axis=-1))
                # class_weights_viz = np.divide(class_weights_viz, np.expand_dims(np.linalg.norm(class_weights_viz, ord=2, axis=-1), axis=-1))


            # features_viz= np.real(features_viz)
            # sc_weights_viz= np.real(sc_weights_viz)
            # class_weights_viz= np.real(class_weights_viz)
            
            fig = plt.figure()
            ax = fig.add_subplot()
            
            for c in np.unique(class_labels):                
                _features = features_viz[c == class_labels]
                ax.scatter(_features[:, 0], _features[:, 1], marker=markers[c],
                           s=50, alpha=0.75, c=np.array([cmap(class_to_sc[c,1] % 13)]))
            
            # PLOT SUPER CLASS CLASSIFIER WEIGHTS 
            # for label in unique_labels:
            #     ax.scatter(sc_weights_viz[unique_labels == label, 0], sc_weights_viz[unique_labels == label, 1], 
            #                     marker='X', s=60, alpha=1, c=np.array([cmap(label)]), label=label)
                
            leg = ax.legend(CATEGORY_NAMES, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.60))
            
            for lh in leg.legendHandles: 
                lh.set_alpha(1)
                
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            # c = np.unique(class_labels)    
            # ax.scatter(class_weights_viz[:, 0], class_weights_viz[:, 1], marker='^',
            #             s=70, alpha=1,  c=np.array([cmap()]), cmap=class_cmap)
            
            if args.do_norm:
                fn = '{:s}_2D_{:s}_norm.png'.format(filename, args.viz)
            else:
                fn = '{:s}_2D_{:s}.png'.format(filename, args.viz)
                
            plt.savefig(os.path.join(args.path_weigths, fn), transparent=False, bbox_inches='tight', dpi=300)
            plt.close()


            
            # 3D embedding
            if args.viz == 'PCA':
                features_viz, _ = pca(features, no_dims=3)
                # sc_weights_pca, _ = pca(sc_weights, no_dims=3)
                # class_weights_pca, _ = pca(class_weights, no_dims=3)
            
                # Other way for PCA
                # sklearn_pca = PCA(n_components=3)
                # sklearn_pca.fit(sc_weights)
                # sc_weights_pca = sklearn_pca.transform(sc_weights)
            
            elif args.viz == 'TSNE':
                tsne = TSNE(3, verbose=1, perplexity= 40, random_state=1234)
                features_viz = tsne.fit_transform(features)
                # sc_weights_viz = tsne.fit_transform(sc_weights)
                # class_weights_viz = tsne.fit_transform(class_weights)
            
            # Normalize features
            if args.do_norm:
                features_viz = np.divide(features_viz, np.expand_dims(np.linalg.norm(features_viz, ord=2, axis=-1), axis=-1))
                # sc_weights_viz = np.divide(sc_weights_viz, np.expand_dims(np.linalg.norm(sc_weights_viz, ord=2, axis=-1), axis=-1))
                # class_weights_viz = np.divide(class_weights_viz, np.expand_dims(np.linalg.norm(class_weights_viz, ord=2, axis=-1), axis=-1))

            # features_pca= np.real(features_pca)
            # sc_weights_pca= np.real(sc_weights_pca)
            # class_weights_pca= np.real(class_weights_pca)
            
            fig = plt.figure(figsize=(12, 12))

            for _ in range(9):

                ax = fig.add_subplot(3, 3, _ + 1, projection='3d')

                # Plot features
            
                for c in np.unique(class_labels):                
                    _features = features_viz[c == class_labels]
                    ax.scatter(_features[:, 0], _features[:, 1], _features[:, 2], marker=markers[c],
                           s=50, alpha=0.75, c=np.array([cmap(class_to_sc[c,1] % 13)]))
                    
                # c = np.unique(sc_labels) 
                # ax.scatter(sc_weights_viz[:, 0], sc_weights_viz[:, 1], sc_weights_viz[:, 2], marker='X',
                #                s=60, alpha=1, c=c, cmap=cmap)
                                        
                # c = np.unique(class_labels) 
                # ax.scatter(class_weights_viz[:, 0], class_weights_viz[:, 1], class_weights_viz[:, 2], marker='X',
                #                s=60, alpha=1, c=c, cmap=class_cmap)
                
                # # Legend
                # leg = ax.legend(CATEGORY_NAMES, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.75))
                # for lh in leg.legendHandles:
                #     lh.set_alpha(1)

                # Plot unitary sphere
                if args.do_norm:
                    x, y, z = create_sphere([0, 0, 0], 0.95)
                    ax.plot_surface(x, y, z, color='gray', edgecolor=None, alpha=0.75)

                # Axes
                if args.do_norm:
                    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                    ax.set_zticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                    ax.axes.set_xlim3d(left=-1.0, right=1.0)
                    ax.axes.set_ylim3d(bottom=-1.0, top=1.0)
                    ax.axes.set_zlim3d(bottom=-1.0, top=1.0)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')

                ax.view_init(10, 30 * _)

            if args.do_norm:
                fn = '{:s}_3D_{:s}_norm.png'.format(filename, args.viz)
            else:
                fn = '{:s}_3D_{:s}.png'.format(filename, args.viz)
            plt.savefig(os.path.join(args.path_weigths, fn),
                transparent=False, bbox_inches='tight', dpi=300)
            plt.close()


def pca(X, no_dims):
    '''
    Computes PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions
    '''
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, :no_dims])
    return Y, M


def create_sphere(center, radius):
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    xyz = np.transpose(np.array([np.ravel(x), np.ravel(y), np.ravel(z)]))
    xyz = np.divide(xyz, np.expand_dims(np.linalg.norm(xyz, ord=2, axis=-1), axis=-1))
    xyz = radius * xyz
    x = center[0] + np.reshape(xyz[:, 0], x.shape)
    y = center[1] + np.reshape(xyz[:, 1], y.shape)
    z = center[2] + np.reshape(xyz[:, 2], z.shape)
    return x, y, z


if __name__ == '__main__':
    main()
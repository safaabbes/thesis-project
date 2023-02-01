import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
import sys

import torch
import torchvision
from sklearn.metrics import confusion_matrix
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image
import cv2
from PIL import Image, ImageDraw, ImageFont 

from datasets import dataset
from models import resnet50_1h, resnet50_2h
sys.path.append('..')
from utils import get_logger, deprocess


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
    parser.add_argument('--gradCAM', default=False, action='store_true')
    
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
    args.test_domain = args_test.test_domain
    args.checkpoint = args_test.checkpoint
    args.path_weights = args_test.path_weights
    args.gradCAM = args_test.gradCAM

    # Create logger
    path_log = os.path.join(args.path_weights, 'log_test_{:s}_{:s}.txt'.format(args.test_domain, args.checkpoint))
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
    logger.info('Model is on device: {}'.format(next(model.parameters()).device))

    # Put model in evaluation mode
    model.eval()

    # Init stats
    running_oa1, running_mca1_num, running_mca1_den = list(), list(), list()
    running_oa2, running_mca2_num, running_mca2_den = list(), list(), list()
    ground_truth_1, ground_truth_2 = list(), list()
    predictions_1, predictions_2 = list(), list()
    final_images = []  
    # Loop over test mini-batches
    for i, data in enumerate(loader_test):
        # Load mini-batch
        images, categories1, categories2 = data
        images = images.to(args.device, non_blocking=True)
        categories1 = categories1.to(args.device, non_blocking=True)
        categories2 = categories2.to(args.device, non_blocking=True)
        ground_truth_1.extend(categories1.tolist())
        ground_truth_2.extend(categories2.tolist())

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
                
            _, preds1 = torch.max(logits1, dim=1)
            predictions_1.extend(preds1.tolist())
            _, preds2 = torch.max(logits2, dim=1)
            predictions_2.extend(preds2.tolist())

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
        
        if args.gradCAM:
            # Apply GradCam to 6 random images
            if i in range(5):
                j=3
                input_tensor = images[j].unsqueeze(0)
                targets = [ClassifierOutputTarget(categories1[j])]
                target_layers = [model.backbone.layer4]
                with GradCAM(model=model, target_layers=target_layers) as cam:
                    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                    img = deprocess(images[j])
                    cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)    
                cam = np.uint8(255*grayscale_cams[0, :])
                cam = cv2.merge([cam, cam, cam])
                imgs = np.hstack((np.uint8(255*img), cam , cam_image))
                final_img = Image.fromarray(imgs)
                draw = ImageDraw.Draw(final_img)
                font = ImageFont.truetype("arial.ttf", 20)
                text = 'Image {}, Correct Label {}, Prediction {}'.format(i*args.bs+j, categories1[j], preds1[j])
                draw.text((10, 10), text, font=font, fill=(255, 0, 0))
                logger.info('Image {}, Correct Label {}, Prediction {}'.format(i*args.bs+j, categories1[j], preds1[j]))
                final_images.append(final_img)
    
    if args.gradCAM:
        final_image = np.concatenate(final_images, axis=0)
        final_image = Image.fromarray(final_image)
        final_image.save(os.path.join(args.path_weights,'gradcam_{}_{}.png'.format(args.test_domain, args.checkpoint)))
    
    # Plot Confusion Matrix
    plot_confusion_matrix(predictions_1, ground_truth_1, args)
    plot_confusion_matrix(predictions_2, ground_truth_2, args)
    
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


def plot_confusion_matrix(predictions, ground_truth, args):
    # Convert to NumPy arrays
    y_true = np.array(ground_truth)
    y_pred = np.array(predictions)
    # Compute confusion matrix
    confusion_mat = confusion_matrix(y_true, y_pred)
    accuracy_matrix = confusion_mat / confusion_mat.sum(axis=1, keepdims=True)
    # Plot confusion matrix using imshow
    plt.figure(figsize=(40, 40))
    plt.imshow(accuracy_matrix, cmap='Blues')
    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix {}_{}'.format(args.test_domain, args.checkpoint))
    # add color bar
    plt.colorbar()
    # Add class numbers as x and y ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(len(accuracy_matrix)))
    ax.set_yticks(np.arange(len(accuracy_matrix)))
    ax.set_xticklabels(np.arange(len(accuracy_matrix)))
    ax.set_yticklabels(np.arange(len(accuracy_matrix)))
    # Add Accuracy text
    for i in range(len(accuracy_matrix)):
        for j in range(len(accuracy_matrix)):
            text = plt.text(j, i, round(accuracy_matrix[i, j], 2),
                        ha="center", va="center", color="black", fontsize=15)

    # save the figure
    if len(accuracy_matrix) == args.num_categories1:
        plt.savefig(os.path.join(args.path_weights,'cm_{}_{}.png'.format(args.test_domain, args.checkpoint)), format='png', dpi=300)
    elif len(accuracy_matrix) == args.num_categories2:
        plt.savefig(os.path.join(args.path_weights,'sc_cm_{}_{}.png'.format(args.test_domain, args.checkpoint)), format='png', dpi=300)
    


if __name__ == '__main__':
    main()
import os
import argparse

from torchvision import transforms 
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim

from utils import *
from datasets import *
from train_backbone import *

dataset_stats = {
  'clipart': ([0.7335894,0.71447897,0.6807669],[0.3542898,0.35537153,0.37871686]),
  'sketch': ([0.8326851 , 0.82697356, 0.8179188 ],[0.25409684, 0.2565908 , 0.26265645]),
  'quickdraw': ([0.95249325, 0.95249325, 0.95249325], [0.19320959, 0.19320959, 0.19320959]) ,
  'real': ([0.6062751 , 0.5892714 , 0.55611473],[0.31526884, 0.3114217 , 0.33154294]) ,
}

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda', help='Computational Device')
    parser.add_argument('--path', type=str, default='/storage/TEV/sabbes/domainnet/', help='Dataset Path')
    parser.add_argument('--source_name', type=str, required=True, help='Source Domain Name')
    parser.add_argument('--target_name', type=str, required=True, help='Target Domain Name')
    parser.add_argument('--bs', type=int, default= 64, help='Batch Size')
    parser.add_argument('--n_epochs', type=int, default= 10, help='Number of Epochs')
    parser.add_argument('--lr', type=float, default= 1e-3, help='Learning Rate')
    
    args=parser.parse_args()
    return args

def main():
    
    args = parse_args()
    
    logger = setup_logger('Logs/sample.log')
    
    for arg, value in vars(args).items():
        logger.info('{} = {}'.format(arg, value))
          
    source_transforms = [transforms.Resize((224, 224)),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize(*dataset_stats[args.source_name]),
                         ]
    target_transforms = [transforms.Resize((224, 224)),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(*dataset_stats[args.target_name]),
                        ]
    
    img_transform_source = transforms.Compose(source_transforms)
    img_transform_target = transforms.Compose(target_transforms)
    
    source_ds = ImageFolder(args.path+args.source_name, img_transform_source)
    target_ds = ImageFolder(args.path+args.target_name, img_transform_target)
    
    logger.info('number of {} samples: {}'.format(args.source_name , len(source_ds)))
    logger.info('number of {} samples: {}'.format(args.target_name , len(target_ds)))
    logger.info('Does the source and target have the same labels?: {}'.format(source_ds.classes == target_ds.classes))
    
    # ONGOING TASK: getting all the stats(mean, std) of dataset to normalize, once each one is done save result in dataset_stats
    # clipart_stats = get_mean_std_dataset(args, 'clipart')                     #DONE
    # sketch_stats = get_mean_std_dataset(args, 'sketch')                       #DONE
    # painting_stats = get_mean_std_dataset(args, 'painting')                   #DATASET STILL NOT UPLOADED ON STORAGE
    # infograph_stats = get_mean_std_dataset(args, 'infograph')                 #DATASET STILL NOT UPLOADED ON STORAGE
    # quickdraw_stats = get_mean_std_dataset(args, 'quickdraw')                 #DONE
    # real_stats = get_mean_std_dataset(args, 'real')                           #DONE
 
    # ONGOING TASK: removing tasks containing <50 samples
    # source_ds = filter_classes(source_ds, logger)
    
    # Create Dataloaders and Split Train / Test
    # Question: in http://ai.bu.edu/M3SDA/ there is a train and test txt file, is that supposed to be used for the split? 
    # if it's the case if we remove some classes would that matter if we followed the split of the txt files anymore?
    s_train, s_test = split_dl(args, source_ds, logger) 
    t_train, t_test = split_dl(args, target_ds, logger) 
    
    #load model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    
    #setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Test and Train for each epoch
    for epoch in range(args.n_epochs):
        train_loss , train_accuracy = train_step(model,s_train, optimizer, logger)
        s_test_loss, s_test_accuracy = test_step(model,s_test, logger)
        t_test_loss, t_test_accuracy = test_step(model,t_test, logger)

        # Log Results
        logger.info('Epoch: {:d}'.format(args.n_epochs+1))
        logger.info('\t Source Train loss {:.5f}, Source Train accuracy {:.2f}'.format(train_loss, train_accuracy))
        logger.info('\t Source Test loss {:.5f}, Source Test accuracy {:.2f}'.format(s_test_loss, s_test_accuracy))
        logger.info('\t Target Test loss {:.5f}, Target Test accuracy {:.2f}'.format(t_test_loss, t_test_accuracy))
        logger.info('-----------------------------------------------------------------------')

if __name__ == '__main__':
    main()
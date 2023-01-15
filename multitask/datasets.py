
import copy
import glob
import json
import os
import numpy as np
from PIL import Image
import time

import torch
import torchvision.transforms as t


class dataset1(torch.utils.data.Dataset):

    def __init__(self, path_pointer, augm_type, domain_type):

        # Get domain flag
        self.domain = 1. if domain_type == 'source' else 0.

        # Get pointers
        pointer = list()
        with open(path_pointer) as f:
            for l in f.readlines():
                pointer.append(l.split())
        self.pointer = np.asarray(pointer)

        # Categories stats
        self.labels1 = [int(category) for category in self.pointer[:, 1]]
        self.instances1 = list()
        for label in np.unique(self.labels1):
            partition = np.where(np.asarray(self.labels1) == label)[0]
            self.instances1.append(len(partition))

        # Super-categories stats
        self.labels2 = [int(category) for category in self.pointer[:, 2]]
        self.instances2 = list()
        for label in np.unique(self.labels2):
            partition = np.where(np.asarray(self.labels2) == label)[0]
            self.instances2.append(len(partition))

        # Data augmentation
        if augm_type == 'train':
            self.transforms = t.Compose([
                t.Resize((256, 256)),
                t.RandomCrop((224, 224)),
                t.RandomHorizontalFlip(p=0.5),
                t.ToTensor(),
                t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        else:
            self.transforms = t.Compose([
                t.Resize((224, 224)),
                t.ToTensor(),
                t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):

        # Load image
        with open(os.path.join('..','..', 'data', self.pointer[index, 0]), 'rb') as _:
            image = Image.open(_)
            image = image.convert('RGB')  # PIL object

        # Apply data augmentation
        image = self.transforms(image)  # PyTorch tensor

        # Load labels
        category1 = torch.tensor(int(self.pointer[index, 1]), dtype=torch.long)  # PyTorch tensor
        category2 = torch.tensor(int(self.pointer[index, 2]), dtype=torch.long)  # PyTorch tensor

        # Load domain flag
        domain = torch.tensor(self.domain, dtype=torch.float)  # PyTorch tensor

        return image, category1, category2, domain

    def __len__(self):
        return len(self.pointer)


class dataset2(torch.utils.data.Dataset):

    def __init__(self, domain_type, augm_type):

        # Get pointers
        pointer = list()
        path_pointer = '../../data/splits_multitask/{:s}_mini.txt'.format(domain_type)
        with open(path_pointer) as f:
            for l in f.readlines():
                pointer.append(l.split())
        self.pointer = np.asarray(pointer)

        # Categories stats
        self.labels1 = [int(category) for category in self.pointer[:, 1]]
        self.instances1 = list()
        for label in np.unique(self.labels1):
            partition = np.where(np.asarray(self.labels1) == label)[0]
            self.instances1.append(len(partition))

        # Super-categories stats
        self.labels2 = [int(category) for category in self.pointer[:, 2]]
        self.instances2 = list()
        for label in np.unique(self.labels2):
            partition = np.where(np.asarray(self.labels2) == label)[0]
            self.instances2.append(len(partition))

        # Data augmentation
        if augm_type == 'train':
            self.transforms = t.Compose([
                t.Resize((256, 256)),
                t.RandomCrop((224, 224)),
                t.RandomHorizontalFlip(p=0.5),
                t.ToTensor(),
                t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        else:
            self.transforms = t.Compose([
                t.Resize((224, 224)),
                t.ToTensor(),
                t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):

        # Load image
        with open(os.path.join('..','..','data', self.pointer[index, 0]), 'rb') as _:
            image = Image.open(_)
            image = image.convert('RGB')  # PIL object

        # Apply data augmentation
        image = self.transforms(image)  # PyTorch tensor

        # Load labels
        category1 = torch.tensor(int(self.pointer[index, 1]), dtype=torch.long)  # PyTorch tensor
        category2 = torch.tensor(int(self.pointer[index, 2]), dtype=torch.long)  # PyTorch tensor

        return image, category1, category2

    def __len__(self):
        return len(self.pointer)
    
    

class PseudoLabelDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels1, labels2, augm_type):
        self.images = images
        self.labels1 = labels1
        self.labels2 = labels2
        
        # Data augmentation
        if augm_type == 'train':
            self.transforms = t.Compose([
                # t.Resize((256, 256)),
                # t.RandomCrop((224, 224)),
                t.RandomHorizontalFlip(p=0.5),
                # Additional Augmentations
                t.RandomHorizontalFlip(p=0.5),
                t.RandomVerticalFlip(p=0.5),
                t.RandomRotation(degrees=90),
                t.GaussianBlur(kernel_size=5),
                # t.ColorJitter(brightness=0.5, hue=0.5), This gives out nan in loss when used
                # t.ToTensor(),
                # t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        else:
            self.transforms = t.Compose([
                t.Resize((224, 224)),
                t.ToTensor(),
                t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = self.transforms(image)
        label1 = self.labels1[index]
        label2 = self.labels2[index]
        return image, label1, label2
import os
import copy
import random
import torch
import numpy as np
from PIL import Image
from collections import Counter
from torch.utils.data import DataLoader, random_split
from torchvision import transforms 
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

from classes import *
from utils import super_class

def find_classes(directory): 
    
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    s1_class_to_idx = {super_cls_name: j for j, super_cls_name in enumerate(s1_classes_dict.keys())}
    s2_class_to_idx = {super_cls_name: j for j, super_cls_name in enumerate(s2_classes_dict.keys())}
    return classes, class_to_idx, s1_class_to_idx, s2_class_to_idx

def make_dataset(directory, class_to_idx = None):
    
    if class_to_idx is None:
        _, class_to_idx, _,_ = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_idx' must have at least one entry to collect any samples.")

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_index
                instances.append(item)
                if target_class not in available_classes:
                    available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        
    return instances

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
    
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        classes, class_to_idx, s1_class_to_idx , s2_class_to_idx= find_classes(self.root)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.s1_class_to_idx = s1_class_to_idx
        self.s2_class_to_idx = s2_class_to_idx
        samples = make_dataset(self.root, class_to_idx)
        self.samples = samples
        self.targets = [super_class(s[1]) for s in samples]
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        target = super_class(target)
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

class DomainNetDataset40():
    def __init__(self, name, img_dir=None):
        self.name = name
        self.img_dir = img_dir
        self.train_data = None
        self.test_data = None  
        self.train_transforms = None
        self.test_transforms = None 
      
    def get_dataset(self):
        
        train_dir = '{}/{}/train'.format(self.img_dir,self.name)
        test_dir = '{}/{}/test'.format(self.img_dir,self.name)
        
        normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        
        self.train_transforms = transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize_transform
                ])

        self.test_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize_transform
            ])
        
        self.train_data = ImageDataset(root=train_dir, 
                                  transform=self.train_transforms, 
                                  ) 

        self.test_data = ImageDataset(root=test_dir, 
                                transform=self.test_transforms)
        
              
        return  self.train_data, self.test_data

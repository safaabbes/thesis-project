import os
import os.path
from PIL import Image
import copy
import random
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from collections import Counter
from torchvision import transforms 
from torchvision.datasets import ImageFolder
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader
import numpy as np

super_classes_dict = { #This was made by using class_to_idx and matching super-classes to classes with the idx
    'animal': ([6, 8, 10, 14, 19, 22, 23, 24, 27, 28, 31, 33, 35], 0), 
    'edible': ([2, 4, 15, 25, 36], 1),
    'transport': ([0, 1, 9, 13, 18, 39], 2),
    'object': ([3, 5, 7, 11, 16, 17, 20, 21, 26, 29, 30, 34, 37, 38], 3),
    'building': ([12, 32], 4),
    } 


def super_class(label: int):  
      
    for key in super_classes_dict.keys():
        if label in super_classes_dict[key][0]: label = (label, super_classes_dict[key][1])
    return label

def find_classes(directory): 
    
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    super_class_to_idx = {super_cls_name: j for j, super_cls_name in enumerate(super_classes_dict.keys())}
    
    return classes, class_to_idx, super_class_to_idx

def make_dataset(directory, class_to_idx = None):
    
    if class_to_idx is None:
        _, class_to_idx, _ = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

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
        classes, class_to_idx, super_class_to_idx = find_classes(self.root)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.super_class_to_idx = super_class_to_idx
        samples = make_dataset(self.root, class_to_idx)
        self.samples = samples
        self.targets = [super_class(s[1]) for s in samples]
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        target = super_class(target)
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            
        # Replace targets by class and super-class labels (exp: 0 label for airplane becomes (0,0) for (airplane,transport))
        # new_targets = add_super_label(self.train_data.targets)
    
        # self.train_data.targets = new_targets
            
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class DomainNetDataset40():
    """
    Dataset Class
    """
    def __init__(self, name, img_dir=None):
        self.name = name
        self.img_dir = img_dir

        self.train_transforms = None
        self.test_transforms = None 
        self.train_size = None
        self.test_size = None
        self.train_data = None
        self.test_data = None    
        
        
    
    def get_dataset(self):
        
        train_dir = '{}/{}/train'.format(self.img_dir,self.name)
        test_dir = '{}/{}/test'.format(self.img_dir,self.name)
        
        normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #taken from sentry but better check this yourself
        
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
        
        self.train_data = ImageDataset(root=train_dir, # target folder of images
                                  transform=self.train_transforms, # transforms to perform on data (images)
                                  ) 

        self.test_data = ImageDataset(root=test_dir, 
                                transform=self.test_transforms)
        
        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)

                                                                               
        return  self.train_data, self.test_data

    def get_dataloaders(self, train_ds, test_ds, num_workers=2, batch_size=128):
        """Constructs and returns dataloaders
        """
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size*2) #increasing bs since the evaluation that requires less computation (No grad)
        
        return train_loader, test_loader



# target_classes = ["airplane", "ambulance", "apple", "backpack", "banana", "bathtub", "bear", "bed", "bee", "bicycle", "bird", "book", "bridge", 
#                 "bus", "butterfly", "cake", "calculator", "camera", "car", "cat", "chair", "clock", "cow", "dog", "dolphin", "donut", "drums", 
#                 "duck", "elephant", "fence", "fork", "horse", "house", "rabbit", "scissors", "sheep", "strawberry", "table", "telephone", "truck"]

# classes = ['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'aircraft_carrier', 'airplane', 'alarm_clock',
#  'ambulance', 'angel', 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana',
#  'bandage', 'barn', 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 
#  'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book',
#  'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 
#  'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 
#  'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cell_phone', 'cello', 'chair', 'chandelier', 
#  'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow',
#  'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 
#  'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser',
#  'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 
#  'flashlight', 'flip_flops', 'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 
#  'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones',
#  'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 
#  'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 'key',
#  'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light_bulb', 'lighter', 'lighthouse', 'lightning', 
#  'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone',
#  'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail',
#  'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint_can', 'paintbrush', 'palm_tree', 'panda',
#  'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup_truck',
#  'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato',
#  'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'rifle', 'river',
#  'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 
#  'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag',
#  'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon',
#  'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign',
#  'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword',
#  'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent', 'tiger', 'toaster',
#  'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle', 
#  'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide', 
#  'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']
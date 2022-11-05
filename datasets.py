import os
import copy
import random
from PIL import Image
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from collections import Counter
from torchvision import transforms 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import utils

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

class UDADataset:
    """
    Dataset Class
    """
    def __init__(self, name, is_target=False, img_dir=None, batch_size=128):
        self.name = name
        self.is_target = is_target
        self.img_dir = img_dir
        self.batch_size = batch_size
  
        self.train_size = None
        self.train_dataset = None
        self.num_classes = None
        self.train_transforms = None
        self.test_transforms = None

    def get_num_classes(self):
        return self.num_classes

    def get_dsets(self):
        """Generates and return train, val, and test datasets
        Returns:
            Train, val, and test datasets.
        """
        data_class = DomainNetDataset(self.name, self.img_dir, self.is_target)

        self.num_classes, self.train_dataset, self.val_dataset, self.test_dataset, self.train_transforms, self.test_transforms = data_class.get_data()

        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_loaders(self, shuffle=True, num_workers=1):
        """Constructs and returns dataloaders
        Args:
            shuffle (bool, optional): Whether to shuffle dataset. Defaults to True.
            num_workers (int, optional): Number of threads. Defaults to 1.
        Returns:
            Train, val, test dataloaders, as well as selected indices used for training
        """
        if not self.train_dataset: self.get_dsets()
  
        num_train = len(self.train_dataset)
        self.train_size = num_train
        
        train_idx = np.arange(len(self.train_dataset))
        
        valid_sampler = SubsetRandomSampler(np.arange(len(self.val_dataset)))		

        train_sampler = SubsetRandomSampler(train_idx)

        train_loader = torch.utils.data.DataLoader(self.train_dataset, sampler=train_sampler, \
                                                   batch_size=self.batch_size, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, sampler=valid_sampler, \
                                                 batch_size=self.batch_size)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)
        return train_loader, val_loader, test_loader, train_idx

class DomainNetDataset:
    """
    DomainNet Dataset class
    """

    def __init__(self, name, img_dir, is_target):
        self.name = name
        self.img_dir = img_dir
        self.is_target = is_target

    def get_data(self):
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

        train_path = os.path.join('data/DomainNet/txt/', '{}_{}_mini.txt'.format(self.name, 'train'))
        test_path = os.path.join('data/DomainNet/txt/', '{}_{}_mini.txt'.format(self.name, 'test'))

        train_dataset = ImageList(open(train_path).readlines(), self.img_dir)
        val_dataset = ImageList(open(test_path).readlines(), self.img_dir)
        test_dataset = ImageList(open(test_path).readlines(), self.img_dir)
        self.num_classes = 40

        train_dataset.targets, val_dataset.targets, test_dataset.targets = torch.from_numpy(train_dataset.labels), \
                                                                           torch.from_numpy(val_dataset.labels), \
                                                                           torch.from_numpy(test_dataset.labels)
        return self.num_classes, train_dataset, val_dataset, test_dataset, self.train_transforms, self.test_transforms

def default_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def make_dataset(image_list, labels=None):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, root, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(image_list)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

        self.root = root
        self.data = np.array([os.path.join(self.root, img[0]) for img in imgs])
        self.labels = np.array([img[1] for img in imgs])
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.data[index], self.labels[index]
        path = os.path.join(self.root, path)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)



























def split_dl(args, ds, logger, test_pct=0.2):
  """
  This function creates train and test DataLoaders.

  """
  test_size = int(test_pct * len(ds))
  train_ds, test_ds = random_split(ds, [len(ds) - test_size, test_size])
  logger.info('Train Size: {}, Test Size {}'.format(len(train_ds), len(test_ds)))

  # PyTorch Data Loaders
  train_dl = DataLoader(train_ds, args.bs, shuffle=True)
  test_dl = DataLoader(test_ds, args.bs*2) #increasing bs since the evaluation that requires less computation (No grad)

  return train_dl, test_dl




classes = ['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'aircraft_carrier', 'airplane', 'alarm_clock',
 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana',
 'bandage', 'barn', 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 
 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book',
 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 
 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 
 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cell_phone', 'cello', 'chair', 'chandelier', 
 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow',
 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 
 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser',
 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 
 'flashlight', 'flip_flops', 'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 
 'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones',
 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 
 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 'key',
 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light_bulb', 'lighter', 'lighthouse', 'lightning', 
 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone',
 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail',
 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint_can', 'paintbrush', 'palm_tree', 'panda',
 'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup_truck',
 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato',
 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'rifle', 'river',
 'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 
 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag',
 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon',
 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign',
 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword',
 'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent', 'tiger', 'toaster',
 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle', 
 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide', 
 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']
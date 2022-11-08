import os
import copy
import random
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from collections import Counter
from torchvision import transforms 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)


class DomainNetDataset40:
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

        self.train_data = ImageFolder(root=train_dir, # target folder of images
                                  transform=self.train_transforms, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

        self.test_data = ImageFolder(root=test_dir, 
                                transform=self.test_transforms)
        
        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)
                                                                                
        return  self.train_data, self.test_data

    def get_dataloaders(self, num_workers=4, batch_size=128):
        """Constructs and returns dataloaders
        """
        if not self.train_data: self.get_dataset()

        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size*2) #increasing bs since the evaluation that requires less computation (No grad)
        # Consider a validation loader in the future
        
        return train_loader, test_loader























# def split_dl(args, ds, logger, test_pct=0.2):
#   """
#   This function creates train and test DataLoaders.

#   """
#   test_size = int(test_pct * len(ds))
#   train_ds, test_ds = random_split(ds, [len(ds) - test_size, test_size])
#   logger.info('Train Size: {}, Test Size {}'.format(len(train_ds), len(test_ds)))

#   # PyTorch Data Loaders
#   train_dl = DataLoader(train_ds, args.bs, shuffle=True)
#   test_dl = DataLoader(test_ds, args.bs*2) #increasing bs since the evaluation that requires less computation (No grad)

#   return train_dl, test_dl




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
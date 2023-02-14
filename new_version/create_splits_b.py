import os
import sys
import numpy as np
sys.path.append('..')
from utils import sorted_alphanumeric

path_input = os.path.join('..','..', 'data', 'splits_baseline')
path_output = os.path.join('..','..', 'data', 'splits_superclass_b')

# DomainNet clustering
CATEGORIES = [
    [0, 'airplane', 11, 'sky_transport'],
    [1, 'ambulance', 5, 'road_transport'],  
    [2, 'apple', 8, 'fruit'],
    [3, 'backpack', 4, 'office'],
    [4, 'banana', 8, 'fruit'],
    [5, 'bathtub', 0, 'furniture'],     
    [6, 'bear', 1, 'mammal'],           
    [7, 'bed', 0, 'furniture'],
    [8, 'bee', 12, 'insect'],
    [9, 'bicycle', 5, 'road_transport'],
    [10, 'bird', 9, 'bird'],
    [11, 'book', 4, 'office'],
    [12, 'bridge', 3, 'building'],
    [13, 'bus', 5, 'road_transport'],
    [14, 'cake', 6, 'food'],
    [15, 'calculator', 2, 'electricity'],
    [16, 'camera', 2, 'electricity'],
    [17, 'car', 5, 'road_transport'],
    [18, 'cat', 1, 'mammal'],
    [19, 'chair', 0, 'furniture'],
    [20, 'clock', 4, 'office'],
    [21, 'cow', 1, 'mammal'],
    [22, 'dog', 1, 'mammal'],
    [23, 'dolphin', 1, 'mammal'],
    [24, 'duck', 9, 'bird'],
    [25, 'elephant', 1, 'mammal'],
    [26, 'fence', 0, 'furniture'],
    [27, 'fork', 10, 'kitchen'],
    [28, 'horse', 1, 'mammal'],
    [29, 'house', 3, 'building'],
    [30, 'rabbit', 1, 'mammal'],
    [31, 'scissors', 4, 'office'],
    [32, 'strawberry', 8, 'fruit'],
    [33, 'table', 0, 'furniture'],
    [34, 'telephone', 2, 'electricity'],
    [35, 'truck', 5, 'road_transport'],
    [36, 'butterfly', 12, 'insect'],  
    [37, 'sheep', 1, 'mammal'],
    [38, 'drums', 7, 'music'],
    [39, 'donut', 6, 'food'],
    ]


CATEGORY_NAMES = ['airplane', 'ambulance', 'apple', 'backpack', 'banana', 'bathtub', 'bear', 'bed',
'bee', 'bicycle', 'bird', 'book', 'bridge', 'bus', 'cake', 'calculator', 'camera',
'car', 'cat', 'chair', 'clock', 'cow', 'dog', 'dolphin', 'duck', 'elephant',
'fence', 'fork', 'horse', 'house', 'rabbit', 'scissors', 'strawberry', 'table',
'telephone', 'truck', 'butterfly', 'sheep', 'drums', 'donut']

SC_CATEGORY_NAMES = ['furniture', 'mammal', 'electricity', 'building', 'office', 'road_transport', 'food', 
                     'music', 'fruit', 'bird', 'kitchen', 'sky_transport', 'insect']

BIAS_CATEGORIES = [
    [0, 'airplane', 4],
    [1, 'ambulance', 3],
    [2, 'apple', 5],
    [3, 'backpack', 5],
    [4, 'banana',  7],
    [5, 'bathtub',  6],
    [6, 'bear',  7],
    [7, 'bed', 4],
    [8, 'bee',  3],
    [9, 'bicycle',  5],
    [10, 'bird', 4],
    [11, 'book',  8],
    [12, 'bridge',  6],
    [13, 'bus',  7],
    [14, 'cake', 0],
    [15, 'calculator',   5],
    [16, 'camera', 7],
    [17, 'car',  2],
    [18, 'cat',  8],
    [19, 'chair', 2],
    [20, 'clock', 3],
    [21, 'cow', 0],
    [22, 'dog',  5],
    [23, 'dolphin', 6],
    [24, 'duck', 2],
    [25, 'elephant', 4],
    [26, 'fence',  0],
    [27, 'fork', 6],
    [28, 'horse', 1],
    [29, 'house', 2],
    [30, 'rabbit',  3],
    [31, 'scissors',  4],
    [32, 'strawberry',  6],
    [33, 'table', 8],
    [34, 'telephone',  1],
    [35, 'truck',  1],
    [36, 'butterfly', 1],  
    [37, 'sheep', 2],
    [38, 'drums', 4],
    [39, 'donut', 8],
    ]


list_files = sorted_alphanumeric(os.listdir(path_input))
for file in list_files:

    if os.path.splitext(file)[0].split('_')[-1] == 'mini':

        path_pointer = os.path.join(path_input, file)
        pointer = list()
        with open(path_pointer) as f:
            for l in f.readlines():
                s1 = l.split()[0]
                s2 = l.split()[1] 
                s3 = str(CATEGORIES[int(s2)][2])
                s4 = str(BIAS_CATEGORIES[int(s2)][2])
                pointer.append([s1, s2, s3, s4])
        pointer = np.asarray(pointer)

        np.savetxt(os.path.join(path_output, file), pointer, fmt='%s')

tmp = np.asarray(CATEGORIES)
categories1 = tmp[:, 0].astype(int)
categories2 = tmp[:, 2].astype(int)
mapping = np.zeros((len(np.unique(categories1)), len(np.unique(categories2))))
for _ in range(len(categories1)):
    mapping[categories1[_], categories2[_]] = 1
np.savez('mapping.npz', data=mapping)

tmp = np.asarray(BIAS_CATEGORIES)
categories1 = tmp[:, 0].astype(int)
categories2b = tmp[:, 2].astype(int)
biased_mapping = np.zeros((len(np.unique(categories1)), len(np.unique(categories2b))))
for _ in range(len(categories1)):
    mapping[categories1[_], categories2b[_]] = 1
np.savez('biased_mapping.npz', data=biased_mapping)
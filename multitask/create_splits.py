
import os
import sys
import numpy as np
sys.path.append('..')
from utils import sorted_alphanumeric

path_input = os.path.join('..','..', 'data', 'splits_baseline')
path_output = os.path.join('..','..', 'data', 'splits_multitask')

# furniture 0 (5 instances)
# mammal 1 (9 instances)
# electricity 2 (3 instances)
# building 3 (2 instances)
# office 4 (4 instances)
# road_transport 5 (5 instances)
# food 6 (2 instances)
# music 7 (only 1)
# fruit 8 (3 instances)
# bird 9 (2 instances)
# kitchen 10 (only 1)
# sky_transport 11 (only 1)
# insect 12 (2 instances)

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

# ERRONEOUS CLASSES DomainNet clustering
# CATEGORIES = [
#     [0, 'airplane', 11, 'sky_transport'],
#     [1, 'ambulance', 5, 'road_transport'],
#     [2, 'apple', 8, 'fruit'],
#     [3, 'backpack', 4, 'office'],
#     [4, 'banana', 8, 'fruit'],
#     [5, 'bathtub', 0, 'furniture'],
#     [6, 'bear', 1, 'mammal'],
#     [7, 'bed', 0, 'furniture'],
#     [8, 'bee', 12, 'insect'],
#     [9, 'bicycle', 5, 'road_transport'],
#     [10, 'bird', 9, 'bird'],
#     [11, 'book', 4, 'office'],
#     [12, 'bridge', 3, 'building'],
#     [13, 'bus', 5, 'road_transport'],
#     [14, 'butterfly', 12, 'insect'],  
#     [15, 'cake', 6, 'food'],
#     [16, 'calculator', 2, 'electricity'],
#     [17, 'camera', 2, 'electricity'],
#     [18, 'car', 5, 'road_transport'],
#     [19, 'cat', 1, 'mammal'],
#     [20, 'chair', 0, 'furniture'],
#     [21, 'clock', 4, 'office'],
#     [22, 'cow', 1, 'mammal'],
#     [23, 'dog', 1, 'mammal'],
#     [24, 'dolphin', 1, 'mammal'],
#     [25, 'donut', 6, 'food'],
#     [26, 'drums', 7, 'music'],
#     [27, 'duck', 9, 'bird'],
#     [28, 'elephant', 1, 'mammal'],
#     [29, 'fence', 0, 'furniture'],
#     [30, 'fork', 10, 'kitchen'],
#     [31, 'horse', 1, 'mammal'],
#     [32, 'house', 3, 'building'],
#     [33, 'rabbit', 1, 'mammal'],
#     [34, 'scissors', 4, 'office'],
#     [35, 'sheep', 1, 'mammal'],
#     [36, 'strawberry', 8, 'fruit'],
#     [37, 'table', 0, 'furniture'],
#     [38, 'telephone', 2, 'electricity'],
#     [39, 'truck', 5, 'road_transport']
#     ]

# Safa's clustering
# CATEGORIES = [
#     [0, 'airplane', 2, 'transport'],
#     [1, 'ambulance', 2, 'transport'],
#     [2, 'apple', 1, 'edible'],
#     [3, 'backpack', 3, 'object'],
#     [4, 'banana', 1, 'edible'],
#     [5, 'bathtub', 0, 'furniture'],
#     [6, 'bear', 0, 'animal'],
#     [7, 'bed', 3, 'object'],
#     [8, 'bee', 0, 'animal'],
#     [9, 'bicycle', 2, 'transport'],
#     [10, 'bird', 0, 'animal'],
#     [11, 'book', 3, 'object'],
#     [12, 'bridge', 4, 'building'],
#     [13, 'bus', 2, 'transport'],
#     [14, 'butterfly', 0, 'animal'],
#     [15, 'cake', 1, 'edible'],
#     [16, 'calculator', 3, 'object'],
#     [17, 'camera', 3, 'object'],
#     [18, 'car', 2, 'transport'],
#     [19, 'cat', 0, 'animal'],
#     [20, 'chair', 3, 'object'],
#     [21, 'clock', 3, 'object'],
#     [22, 'cow', 0, 'animal'],
#     [23, 'dog', 0, 'animal'],
#     [24, 'dolphin', 0, 'animal'],
#     [25, 'donut', 1, 'edible'],
#     [26, 'drums', 3, 'object'],
#     [27, 'duck', 0, 'animal'],
#     [28, 'elephant', 0, 'animal'],
#     [29, 'fence', 3, 'object'],
#     [30, 'fork', 3, 'object'],
#     [31, 'horse', 0, 'animal'],
#     [32, 'house', 4, 'building'],
#     [33, 'rabbit', 1, 'mammal'],
#     [34, 'scissors', 3, 'object'],
#     [35, 'sheep', 0, 'animal'],
#     [36, 'strawberry', 1, 'edible'],
#     [37, 'table', 3, 'object'],
#     [38, 'telephone', 3, 'object'],
#     [39, 'truck', 2, 'transport']
#     ]

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
                pointer.append([s1, s2, s3])
        pointer = np.asarray(pointer)

        np.savetxt(os.path.join(path_output, file), pointer, fmt='%s')

tmp = np.asarray(CATEGORIES)
categories1 = tmp[:, 0].astype(int)
categories2 = tmp[:, 2].astype(int)
mapping = np.zeros((len(np.unique(categories1)), len(np.unique(categories2))))
for _ in range(len(categories1)):
    mapping[categories1[_], categories2[_]] = 1
np.savez('mapping.npz', data=mapping)
################################################################################################################
#### 40 Main Classes
################################################################################################################

main_classes = [
    "airplane", "ambulance", "apple", "backpack", "banana", "bathtub", "bear", "bed", "bee", "bicycle", "bird", "book", "bridge", 
    "bus", "butterfly", "cake", "calculator", "camera", "car", "cat", "chair", "clock", "cow", "dog", "dolphin", "donut", "drums", 
    "duck", "elephant", "fence", "fork", "horse", "house", "rabbit", "scissors", "sheep", "strawberry", "table", "telephone", "truck"
    ]

class_to_idx = {
    'airplane': 0, 'ambulance': 1, 'apple': 2, 'backpack': 3, 'banana': 4, 'bathtub': 5, 
    'bear': 6, 'bed': 7, 'bee': 8, 'bicycle': 9, 'bird': 10, 'book': 11, 'bridge': 12, 'bus': 13, 'butterfly': 14, 'cake': 15, 
    'calculator': 16, 'camera': 17, 'car': 18, 'cat': 19, 'chair': 20, 'clock': 21, 'cow': 22, 'dog': 23, 'dolphin': 24, 
    'donut': 25, 'drums': 26, 'duck': 27, 'elephant': 28, 'fence': 29, 'fork': 30, 'horse': 31, 'house': 32, 'rabbit': 33,
    'scissors': 34, 'sheep': 35, 'strawberry': 36, 'table': 37, 'telephone': 38, 'truck': 39
    }  

idx_to_class = {
    0: 'airplane', 1: 'ambulance', 2: 'apple', 3: 'backpack', 4: 'banana', 5: 'bathtub', 
    6: 'bear', 7: 'bed', 8: 'bee', 9: 'bicycle', 10: 'bird', 11: 'book', 12: 'bridge', 13: 'bus', 14: 'butterfly', 15: 'cake', 
    16: 'calculator', 17: 'camera', 18: 'car', 19: 'cat', 20: 'chair', 21: 'clock', 22: 'cow', 23: 'dog', 24: 'dolphin', 
    25: 'donut', 26: 'drums', 27: 'duck', 28: 'elephant', 29: 'fence', 30: 'fork', 31: 'horse', 32: 'house', 33: 'rabbit',
    34: 'scissors', 35: 'sheep', 36: 'strawberry', 37: 'table', 38: 'telephone', 39: 'truck'
    }

################################################################################################################
#### Personal Clustering (Super Class 1)
################################################################################################################

s1_classes = ["animal", "edible", "transport", "object", "building"]

s1_classes_dict = { #This was made by using class_to_idx and matching super-classes to classes with the idx
    'animal': ([6, 8, 10, 14, 19, 22, 23, 24, 27, 28, 31, 33, 35], 0), 
    'edible': ([2, 4, 15, 25, 36], 1),
    'transport': ([0, 1, 9, 13, 18, 39], 2),
    'object': ([3, 5, 7, 11, 16, 17, 20, 21, 26, 29, 30, 34, 37, 38], 3),
    'building': ([12, 32], 4),
    } 

sorted_s1_classes = [
    'bear', 'bee', 'bird', 'butterfly', 'cat', 'cow', 'dog', 'dolphin', 'duck', 'elephant', 'horse', 'rabbit', 'sheep',
    'apple', 'banana', 'cake', 'donut', 'strawberry',
    'airplane', 'ambulance', 'bicycle', 'bus', 'car', 'truck',
    'backpack', 'bathtub', 'bed', 'book', 'calculator', 'camera', 'chair', 'clock', 'drums', 'fence', 'fork', 'scissors', 'table', 'telephone',
    'bridge', 'house']

sorted_s1_classes_idx = [
    6, 8, 10, 14, 19, 22, 23, 24, 27, 28, 31, 33, 35,  
    2, 4, 15, 25, 36,
    0, 1, 9, 13, 18, 39,
    3, 5, 7, 11, 16, 17, 20, 21, 26, 29, 30, 34, 37, 38,
    12, 32
    ]

################################################################################################################
#### DomainNet Clustering (Super Class 2)
################################################################################################################

s2_classes = ['mammal', 'bird', 'insect', 'fruit', 'food', 'road_transport', 'sky_transport', 'furniture', 'electricity', 'office', 'kitchen', 'music', 'building' ]

s2_classes_dict = { #This was made by using DomainNet Paper Clustering
    'mammal': ([6, 19, 22, 23, 24, 28, 31, 33, 35], 0), 
    'bird': ([27, 10], 1) ,
    'insect': ([8, 14], 2) ,
    'fruit': ([2, 4, 36], 3),
    'food': ([15, 25], 4) ,
    'road_transport': ([1, 9, 13, 18, 39], 5),
    'sky_transport': ([0], 6) ,
    'furniture': ([5, 7, 20, 29, 37], 7),
    'electricity': ([16, 17, 38], 8) ,
    'office': ([11, 34, 21, 3], 9) ,
    'kitchen': ([30], 10) ,
    'music':  ([26], 11) ,
    'building': ([12, 32], 12),
    } 

sorted_s2_classes = [
    'bear','cat', 'cow', 'dog', 'dolphin', 'elephant', 'horse', 'rabbit', 'sheep',
    'bird','duck', 
    'bee','butterfly',
    'apple', 'banana', 'strawberry',
    'cake', 'donut',
    'ambulance', 'bicycle', 'bus', 'car', 'truck',
    'airplane', 
    'bathtub', 'bed', 'chair', 'fence', 'table', 
    'calculator','camera', 'telephone', 
    'backpack', 'book',  'clock', 'scissors', 
    'fork', 
    'drums',
    'bridge', 'house'
]

sorted_s2_classes_idx = [
    6, 19, 22, 23, 24, 28, 31, 33, 35,  
    10, 27,
    8, 14,
    2, 4, 36,
    15, 25,
    1, 9, 13, 18, 39,
    0, 
    5, 7, 20, 29, 37,
    16, 17, 38,
    3, 11, 21, 34,
    30,
    26,
    12, 32
    ]

################################################################################################################
#### Original Classes
################################################################################################################

CATEGORY_NAMES = [
    'The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'aircraft_carrier', 'airplane', 'alarm_clock',
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
    'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag'
    ]
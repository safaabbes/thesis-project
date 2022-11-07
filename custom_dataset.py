# Here we present the code used to create domainnet40 dataset 
# The dataset is based on the COAL/SENTRY research papers. it contains 4 domain (clipart, real, sketch, painting) and 40 classes per domain. 
# The sampling made is derived from the SENTRY code by using their split to create the dataset (https://github.com/virajprabhu/SENTRY/tree/main/data/DomainNet/txt)
# target_classes = ["airplane", "ambulance", "apple", "backpack", "banana", "bathtub", "bear", "bed", "bee", "bicycle", "bird", "book", "bridge", 
#                 "bus", "butterfly", "cake", "calculator", "camera", "car", "cat", "chair", "clock", "cow", "dog", "dolphin", "donut", "drums", 
#                 "duck", "elephant", "fence", "fork", "horse", "house", "rabbit", "scissors", "sheep", "strawberry", "table", "telephone", "truck"]

import pathlib
import shutil

domains = ['clipart', 'real', 'sketch', 'painting']
data_splits=["train", "test"]

class_splits = {}
full_data = {}

for domain in domains:
    for data_split in data_splits:
        # Get data
        label_path = "/scratch/TEV/sabbes/src/data/DomainNet/txt/{}_{}_mini.txt".format(domain, data_split)
        with open(label_path, "r") as f:
            data_list = [line.strip("\n") for line in f.readlines()] 
        # remove the assigned label integer 
        separator = '.jpg'
        data = [text.split(separator, 1)[0] + separator for text in data_list]    
        # Apply full paths
        image_paths = [pathlib.Path(data_item) for data_item in data]
        class_splits[data_split] = image_paths

    target_dir_name = "/storage/TEV/sabbes/domainnet40/{}".format(domain)
    print(f"Creating directory: '{target_dir_name}'")

    # Setup the directories
    target_dir = pathlib.Path(target_dir_name)

    # Make the directories
    target_dir.mkdir(parents=True, exist_ok=True)
        for image_split in class_splits.keys():
            for image_path in class_splits[str(image_split)]:
                source_path = '/storage/TEV/sabbes/domainnet/' + str(image_path)
                dest_dir = target_dir / image_split / image_path.parent.stem / image_path.name
                if not dest_dir.parent.is_dir():
                    dest_dir.parent.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] Copying {image_path} to {dest_dir}...")
                shutil.copy2(source_path, dest_dir)
    class_splits = {}     
    
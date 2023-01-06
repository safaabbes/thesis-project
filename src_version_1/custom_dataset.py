import pathlib
import shutil

domains = ['clipart', 'real', 'sketch', 'painting']
data_splits=["train", "test"]

class_splits = {}
full_data = {}

for domain in domains:
    for data_split in data_splits:
        # Get data
        label_path = "../../data/splits_baseline/{}_{}_mini.txt".format(domain, data_split)
        with open(label_path, "r") as f:
            data_list = [line.strip("\n") for line in f.readlines()] 
        # remove the assigned label integer 
        separator = '.jpg'
        data = [text.split(separator, 1)[0] + separator for text in data_list]    
        # Apply full paths
        image_paths = [pathlib.Path(data_item) for data_item in data]
        class_splits[data_split] = image_paths

    target_dir_name = "../../data/domainnet40/{}".format(domain)
    print(f"Creating directory: '{target_dir_name}'")

    # Setup the directories
    target_dir = pathlib.Path(target_dir_name)

    # Make the directories
    target_dir.mkdir(parents=True, exist_ok=True)
    for image_split in class_splits.keys():
        for image_path in class_splits[str(image_split)]:
            source_path = '../../data/' + str(image_path)
            dest_dir = target_dir / image_split / image_path.parent.stem / image_path.name
            if not dest_dir.parent.is_dir():
                dest_dir.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Copying {image_path} to {dest_dir}...")
            shutil.copy2(source_path, dest_dir)
    class_splits = {}     
    
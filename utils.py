import logging
from torchvision import transforms 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np

def setup_logger(file_name=None, logger_name = __name__): 
    """ 	
    function to set logger

    """
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers.clear()
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter) 
    logger.addHandler(sh)
    
    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
       
     
    return logger



def get_mean_std_dataset(args, name_ds):
    """ 	
    function to compute the mean and standard deviation across each channel of the whole dataset

    """
    root_dir = args.path+ name_ds    #dataset path
    
    transforms_list = [transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
    ]


    dataset = ImageFolder(root=root_dir, transform=transforms.Compose(transforms_list))
    data_loader = DataLoader(dataset, batch_size=800,shuffle=False) 	# set large batch size to get good approximate of mean, std of full dataset

    mean = []
    std = []

    for i, data in enumerate(data_loader):
        # shape is (batch_size, channels, height, width)
        npy_image = data[0].numpy()

        # compute mean, std per batch shape (3,) three channels
        batch_mean = np.mean(npy_image, axis=(0,2,3))
        batch_std = np.std(npy_image, axis=(0,2,3))

        mean.append(batch_mean)
        std.append(batch_std)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    mean = np.array(mean).mean(axis=0) # average over batch averages
    std = np.array(std).mean(axis=0) # average over batch stds
    return mean , std

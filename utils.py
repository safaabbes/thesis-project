import logging
import random
import numpy as np
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import itertools
import seaborn as sns
from sklearn.metrics import confusion_matrix
cluster_sorted_labels = [6, 8, 10, 14, 19, 22, 23, 24, 27, 28, 31, 33, 35,   #There is no label 1000 so it would be a null line to seperate
                         2, 4, 15, 25, 36,
                         0, 1, 9, 13, 18, 39,
                         3, 5, 7, 11, 16, 17, 20, 21, 26, 29, 30, 34, 37, 38,
                         12, 32]

cluster_sorted_classes = ['bear', 'bee', 'bird', 'butterfly', 'cat', 'cow', 'dog', 'dolphin', 'duck', 'elephant', 'horse', 'rabbit', 'sheep',
           'apple', 'banana', 'cake', 'donut', 'strawberry',
           'airplane', 'ambulance', 'bicycle', 'bus', 'car', 'truck',
           'backpack', 'bathtub', 'bed', 'book', 'calculator', 'camera', 'chair', 'clock', 'drums', 'fence', 'fork', 'scissors', 'table', 'telephone',
           'bridge', 'house']

dataset_stats = {
  'clipart': ([0.7335894,0.71447897,0.6807669],[0.3542898,0.35537153,0.37871686]),
  'sketch': ([0.8326851 , 0.82697356, 0.8179188 ],[0.25409684, 0.2565908 , 0.26265645]),
  'quickdraw': ([0.95249325, 0.95249325, 0.95249325], [0.19320959, 0.19320959, 0.19320959]) ,
  'real': ([0.6062751 , 0.5892714 , 0.55611473],[0.31526884, 0.3114217 , 0.33154294]) ,
}

######################################################################
##### Train/Test utilities
######################################################################

def get_num_correct(preds, labels):
  return preds.argmax(dim=1).eq(labels).sum().item()

def test_step(args, model, data_loader, logger):
  nb_samples = 0
  cumulative_loss = 0.
  total_correct = 0
  all_preds = torch.tensor([], dtype = torch.int).to(args.device) #for the confusion matrix
  all_true = torch.tensor([], dtype = torch.int).to(args.device) #for the confusion matrix
  # set the network to evaluation mode
  model.eval() 
  logger.info("Start Testing")
  # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
  with torch.no_grad():
    # iterate over the test set
    for batch in data_loader:
      # logger.info('Test Batch {} out of {}'.format(batch_idx+1, len(data_loader)))
      data, (labels, _ ) = batch
      # forward pass
      preds = model(data)
      # Compute loss
      loss = F.cross_entropy(preds,labels)
      # update cumulative values
      nb_samples += data.shape[0]
      cumulative_loss += loss.item()
      total_correct += get_num_correct(preds, labels)
      # Save predictions
      all_preds = torch.cat((all_preds, preds.argmax(dim=1).int()), dim=0)
      all_true = torch.cat((all_true, labels.int()), dim=0)
  
    # Compute average loss
    average_loss = cumulative_loss / nb_samples
    # Compute average accuracy
    average_accuracy = total_correct / nb_samples * 100
    # Confusion Matrix
    all_preds = all_preds.tolist()
    all_true = all_true.tolist()
    cm_acc = confusion_matrix( all_true, all_preds, labels = cluster_sorted_labels,  normalize='true')
    # cm = np.zeros((40,40), dtype = np.int)
    # for i,j in zip(all_true, all_preds,):
    #   cm[i,j] += 1
    # Create Confusion Matrix for accuracy
    # acc_cm = np.zeros((40,40), dtype = np.float)  
    # for i in range(40):
    #   for j in range(40):
    #     acc_value = cm[i,j] / cm.sum(axis=1)[i] * 100
    #     acc_cm[i,j] = round(acc_value,2)
    # Compute Per-Class Average Accuracy (Used in COAL PAPER)
    per_cls_acc_vec = cm_acc.diagonal() / cm_acc.sum(axis=1) * 100  
    # print('per_cls_acc_vec: ', per_cls_acc_vec)
    per_cls_avg_acc = per_cls_acc_vec.mean()
    # per_cls_acc_list = { i: np.round(per_cls_acc_vec[i], 2) for i in range(len(per_cls_acc_vec))}
    # per_cls_samples = { i: cm[i,:].sum() for i in range(len(target_classes))}

  return average_loss, average_accuracy, per_cls_avg_acc, cm_acc

######################################################################
##### Optimization utilities
######################################################################

def generate_optimizer(model, args):
	param_list = []
	lr = args.lr 
	wd = args.wd 
	if args.optimizer == 'Adam':
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
	elif args.optimizer == 'SGD':
		for key, value in dict(model.named_parameters()).items():
			if value.requires_grad:
				if 'classifier' not in key:
					param_list += [{'params': [value], 'lr': lr*0.1, 'weight_decay': wd}]
				else:
					param_list += [{'params': [value], 'lr': lr, 'weight_decay': wd}]

		optimizer = optim.SGD(param_list, momentum=0.9, weight_decay=wd, nesterov=True)
	else: raise NotImplementedError
	return optimizer

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=0.0001, power=0.75, init_lr=0.001):
	"""
	Decay learning rate
	"""	
	for i, param_group in enumerate(optimizer.param_groups):
		lr = param_lr[i] * (1 + gamma * iter_num) ** (- power)
		param_group['lr'] = lr
	return optimizer

######################################################################
##### Logging utilities
######################################################################

def setup_logger(file_name=None, logger_name = __name__): 
        
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

######################################################################
##### Moving to GPU utilities
######################################################################

class DeviceDataLoader():
    """
    Wrap a dataloader to move data to a device
    
    """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        self.dataset = dl.dataset
        

        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device) #The yield keyword in Python is used to create a generator function that can be used within a for loop,

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def to_device(data, device):
    """
    Move tensors to chosen device
    
    """
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

######################################################################
##### Special Tasks utilities
######################################################################

def filter_classes(ds, logger):
  count = dict(Counter(ds.targets))
  
  filtered_dict = dict()
  classes = []
  # Iterate over all the items in dictionary and filter items <50
  for (key, value) in count.items():
    # Check if key is even then add pair to new dictionary
    if value>=50:
        filtered_dict[key] = value
        classes.append(ds.classes[key])     
  
  logger.info('{} remained classes: {}'.format(len(classes), classes)) 
  #We have a list of the classes that have more than 50 samples and a dictionary of the index , nb of samples per class
  
  #We need to re-create a dataset (?) What would be the best way to remove a class from the dataset already created ?
  # (implement custom functions on ImageFolder or DatasetFolder library?)
  
  # another problem is that we need to select the common remained classes over all the domains before re-creating the classes
  # for that we could apply this method on every domain and create a final classes list with the intersections of the list
  
  return classes

def get_mean_std_dataset(args, name_ds):
    """ 	
    function to compute the mean and standard deviation across each channel of the whole dataset

    """
    root_dir = args.path+ name_ds    #dataset path
    
    transforms_list=[transforms.Resize((224, 224)),
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


def plot_confusion_matrix(ax, fig, cm,
                          title='Confusion matrix',
                        ):
  
  sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=cluster_sorted_classes, yticklabels=cluster_sorted_classes, cmap=plt.cm.Blues, linewidths=1, linecolor='white')
  y_labels = ax.get_yticklabels()
  y_ticks = ax.get_yticks()
  x_labels = ax.get_xticklabels()
  for xlabel, ylabel , tick in zip(x_labels, y_labels, y_ticks):
    if tick < 13:
      xlabel.set_color('r')
      ylabel.set_color('r')
    elif (tick > 13 and tick < 18): 
      xlabel.set_color('b')
      ylabel.set_color('b')
    elif (tick > 18 and tick < 24):   
      xlabel.set_color('g')
      ylabel.set_color('g')
    elif (tick > 24 and tick < 38):   
      xlabel.set_color('m')
      ylabel.set_color('m')
    elif tick > 38:
      xlabel.set_color('y')
      ylabel.set_color('y')
  plt.ylabel('Ground Truth')
  plt.xlabel('Predicted')
 



def count_nb_per_class(ds , labels):
    dictionary = dict(Counter(ds.targets))
    dict2 = dict()
    for x in dictionary:
        (y,_) = x
        dict2[y] = dictionary[x]    
        
    data = [[label, val] for (label, val) in zip(labels, dictionary.values())]
    table = wandb.Table(data=data, columns = ["Classes", "Number of Samples"])
    wandb.log({"DS" : wandb.plot.bar(table, "Classes",
                                "Number of Samples", title="DS Set")})
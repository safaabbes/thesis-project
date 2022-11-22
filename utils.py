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
      total_correct += get_num_correct(preds, labels)
      # Save predictions
      all_preds = torch.cat((all_preds, preds.argmax(dim=1).int()), dim=0)
      all_true = torch.cat((all_true, labels.int()), dim=0)
  
    # Compute average accuracy
    average_accuracy = total_correct / nb_samples * 100
    # Confusion Matrix
    all_preds = all_preds.tolist()
    all_true = all_true.tolist()
    cm = np.zeros((40,40), dtype = np.int)
    for i,j in zip(all_true, all_preds,):
      cm[i,j] += 1
    # Compute Per-Class Average Accuracy (Used in COAL PAPER)
    per_cls_acc_vec = cm.diagonal() / cm.sum(axis=1) * 100  
    per_cls_avg_acc = per_cls_acc_vec.mean()
    # per_cls_acc_list = { i: np.round(per_cls_acc_vec[i], 2) for i in range(len(per_cls_acc_vec))}
    # per_cls_samples = { i: cm[i,:].sum() for i in range(len(target_classes))}

  return average_accuracy, per_cls_avg_acc, cm

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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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
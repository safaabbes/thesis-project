import os
import logging
import random
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import itertools
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb

from classes import *

######################################################################
##### Dataset utilities
######################################################################

def super_class(label):  
  for key1 in s1_classes_dict.keys():
    if label in s1_classes_dict[key1][0]: 
      for key2 in s2_classes_dict.keys():
        if label in s2_classes_dict[key2][0]: label = (label, s1_classes_dict[key1][1], s2_classes_dict[key2][1])
  return label

######################################################################
##### Train/Test utilities
######################################################################

def get_num_correct(preds, labels):
  return preds.argmax(dim=1).eq(labels).sum().item()

def test_step(args, model, data_loader, criterion, logger):
  running_oa1, running_mca1_num, running_mca1_den = list(), list(), list()
  running_oa2, running_mca2_num, running_mca2_den = list(), list(), list()
  model.eval() 
  logger.info("Start Testing")
  
  for data in data_loader:
    images, (s_labels, s1_labels, s2_labels) = data
    with torch.inference_mode():
      # Forward pass
      if args.model_type == 'R50_2H':
        logits1, logits2 = model(images)
        _, preds1 = torch.max(logits1, dim=1)
        _, preds2 = torch.max(logits2, dim=1)
      elif args.model_type == 'R50_1H':
        logits1 = model(images)
        _, preds1 = torch.max(logits1, dim=1)
        
        tmp = np.load('mapping.npz')
        mapping = torch.tensor(tmp['data'], dtype=torch.float32, device=args.device, requires_grad=False)
        logits2 = torch.mm(logits1, mapping) / (1e-6 + torch.sum(mapping, dim=0))
        _, preds2 = torch.max(logits2, dim=1)
    
    # Update metrics
    oa1 = torch.sum(preds1 == s_labels.squeeze()) / len(s_labels)
    running_oa1.append(oa1.item())
    mca1_num = torch.sum(
        torch.nn.functional.one_hot(preds1, num_classes=40) * \
        torch.nn.functional.one_hot(s_labels, num_classes=40), dim=0)
    mca1_den = torch.sum(
        torch.nn.functional.one_hot(s_labels, num_classes=40), dim=0)
    running_mca1_num.append(mca1_num.detach().cpu().numpy())
    running_mca1_den.append(mca1_den.detach().cpu().numpy())

    oa2 = torch.sum(preds2 == s2_labels.squeeze()) / len(s2_labels)
    running_oa2.append(oa2.item())
    mca2_num = torch.sum(
        torch.nn.functional.one_hot(preds2, num_classes=13) * \
        torch.nn.functional.one_hot(s2_labels, num_classes=13), dim=0)
    mca2_den = torch.sum(
        torch.nn.functional.one_hot(s2_labels, num_classes=13), dim=0)
    running_mca2_num.append(mca2_num.detach().cpu().numpy())
    running_mca2_den.append(mca2_den.detach().cpu().numpy())

  # Update MCA metric
  mca1_num = np.sum(running_mca1_num, axis=0)
  mca1_den = 1e-16 + np.sum(running_mca1_den, axis=0)
  mca2_num = np.sum(running_mca2_num, axis=0)
  mca2_den = 1e-16 + np.sum(running_mca2_den, axis=0)

  stats = {
      'oa1': np.mean(running_oa1),
      'mca1': np.mean(mca1_num/mca1_den),
      'oa2': np.mean(running_oa2),
      'mca2': np.mean(mca2_num/mca2_den),
      }
    
  return stats
######################################################################
##### Plotting utilities
######################################################################

def plot_confusion_matrix(ax, fig, cm):
  sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=sorted_s1_classes, yticklabels=sorted_s1_classes, 
              cmap=plt.cm.Blues, linewidths=1, linecolor='white')
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

def plot_samples_per_cls(instances_list):
  labels = main_classes
  values = instances_list
  data = [[label, val] for (label, val) in zip(labels, values)]
  table = wandb.Table(data=data, columns = ["Classes", "Number of Samples"])
  wandb.log({"DS" : wandb.plot.bar(table, "Classes", "Number of Samples", title="DS Set")})
  
  # fig, ax = plt.subplots(figsize=(50,50))
  # x = list(range(40))  
  # plt.rcParams.update({'font.size': 22})
  # plt.barh(x, y, tick_label = tick_label, color = ['red', 'green']) 
  # for index, value in enumerate(y): plt.text(value, index, str(value))
  # plt.title('Instances of labels over 20 mini-batches with balance = {}'.format(args.cbt))
  # wandb.log({'bar': wandb.Image(plt)})

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
##### Moving to GPU utilities
######################################################################

class DeviceDataLoader():
  def __init__(self, dl, device):
    self.dl = dl
    self.device = device
    self.dataset = dl.dataset
      
  def __iter__(self):
    for b in self.dl: 
      yield to_device(b, self.device) 

  def __len__(self):
    return len(self.dl)

def to_device(data, device):
  if isinstance(data, (list,tuple)):
    return [to_device(x, device) for x in data]
  return data.to(device, non_blocking=True)

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
##### Special Task utilities
######################################################################

def count_nb_per_class(ds , labels):
  dictionary = dict(Counter(ds.targets))
  dict2 = dict()
  for x in dictionary:
    (y,_) = x
    dict2[y] = dictionary[x]        
  data = [[label, val] for (label, val) in zip(labels, dictionary.values())]
  table = wandb.Table(data=data, columns = ["Classes", "Number of Samples"])
  wandb.log({"DS" : wandb.plot.bar(table, "Classes", "Number of Samples", title="DS Set")})
    

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

def test_step(args, model, data_loader, logger):
  nb_samples = 0
  cumulative_loss = 0.
  total_correct = 0
  all_preds = torch.tensor([], dtype = torch.int).to(args.device) 
  all_true = torch.tensor([], dtype = torch.int).to(args.device)
  s1_all_true = torch.tensor([], dtype = torch.int).to(args.device)
  s2_all_true = torch.tensor([], dtype = torch.int).to(args.device)
  model.eval() 
  logger.info("Start Testing")
  with torch.no_grad():
    for batch in data_loader:
      data, (labels, s1_labels, s2_labels) = batch
      preds = model(data)
      loss = F.cross_entropy(preds,labels)
      # update cumulative values
      nb_samples += data.shape[0]
      cumulative_loss += loss.item()
      total_correct += get_num_correct(preds, labels)
      # Save predictions
      all_preds = torch.cat((all_preds, preds.argmax(dim=1).int()), dim=0)
      all_true = torch.cat((all_true, labels.int()), dim=0)
      # Save Super Labels Ground Truth
      s1_all_true = torch.cat((s1_all_true, s1_labels.int()), dim=0)
      s2_all_true = torch.cat((s2_all_true, s2_labels.int()), dim=0)
      
    # Save Super Labels 1 Predictions
    s1_preds_list = [super_class(label)[1] for label in all_preds]
    s1_all_preds = torch.tensor(s1_preds_list, dtype = torch.int).to(args.device)
    # Save Super Labels 2 Predictions
    s2_preds_list = [super_class(label)[2] for label in all_preds]
    s2_all_preds = torch.tensor(s2_preds_list, dtype = torch.int).to(args.device)
    # Compute average loss
    average_loss = cumulative_loss / nb_samples
    # Compute average accuracy
    average_accuracy = total_correct / nb_samples * 100
    # Confusion Matrix
    all_preds = all_preds.tolist()
    all_true = all_true.tolist()
    cm_acc = confusion_matrix( all_true, all_preds, labels = sorted_s1_classes_idx,  normalize='true')
    # Super Classes 1 Confusion Matrix
    s1_all_true = s1_all_true.tolist()
    s1_all_preds = s1_all_preds.tolist()
    s1_cm_acc = confusion_matrix( s1_all_true, s1_all_preds, labels = list(range(5)),  normalize='true')
    # Super Classes 2 Confusion Matrix
    s2_all_true = s2_all_true.tolist()
    s2_all_preds = s2_all_preds.tolist()
    s2_cm_acc = confusion_matrix( s2_all_true, s2_all_preds, labels = list(range(13)),  normalize='true')
    # Compute average accuracy
    per_cls_avg_acc = cm_acc.diagonal().mean() * 100  

  return average_loss, average_accuracy, per_cls_avg_acc, cm_acc, s1_cm_acc, s2_cm_acc

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
    

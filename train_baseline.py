import os
import sys
import random
import torch
import time
import datetime
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import wandb
from sklearn.metrics import confusion_matrix
from collections import Counter
import seaborn as sns

from utils import *
from classes import *

def train_baseline( s_train_dl, s_test_dl, t_train_dl, t_test_dl, model, args, optimizer,  logger):  
  # Setting Wandb
  wandb.init(
      project='Baseline-SENTRY-Train',
      name=args.exp,
      config = {
                "source": args.source,
                "target": args.target,
                "lr": args.lr,
                "epochs": args.n_epochs,
                "batch_size": args.bs,
                })

  #set starting time
  since = time.time()
  for epoch in range(args.n_epochs):
    train_loss , train_accuracy = source_train_step(epoch, args, model,s_train_dl, optimizer, logger)
    # Testing
    s_loss, s_test_accuracy, s_per_cls_avg_acc, s_cm, s1_s_cm, s2_s_cm= test_step(args, model,s_test_dl, logger)
    t_loss, t_test_accuracy, t_per_cls_avg_acc, t_cm, s1_t_cm, s2_t_cm = test_step(args, model,t_test_dl, logger)
    # Log Results
    logger.info('Epoch: {:d}'.format(epoch+1))
    logger.info('\t Source Train loss {:.5f}, Source Train accuracy {:.2f}'.format(train_loss, train_accuracy))
    logger.info('\t Source Test accuracy {:.2f}, Source per_cls_avg_acc {:.2f}'.format(s_test_accuracy, s_per_cls_avg_acc))
    logger.info('\t Target Test accuracy {:.2f}, Target per_cls_avg_acc {:.2f}'.format(t_test_accuracy, t_per_cls_avg_acc))
    logger.info('-----------------------------------------------------------------------')
    # Log results to Wandb
    metrics = {#Train Results
               "train/train_loss": train_loss, 
               "train/train_acc": train_accuracy,
               #Source Test Results
               "s_test/test_per_cls_avg_acc":s_per_cls_avg_acc,
               "s_test/test_avg_acc":s_test_accuracy,
               #Target Test Results
               "t_test/test_per_cls_avg_acc":t_per_cls_avg_acc,
               "t_test/test_avg_acc":t_test_accuracy,
            }
    wandb.log({**metrics})
    
  # Log time
  duration = time.time() - since
  logger.info('Training duration: {}'.format(str(datetime.timedelta(seconds=duration))))
# Plot Confusion Matrix
  fig, ax = plt.subplots(figsize=(50,50))
  plot_confusion_matrix(ax, fig, s_cm)
  wandb.log({"s_test/s_cm": wandb.Image(plt)})
  plt.close()
  fig, ax = plt.subplots(figsize=(50,50))
  plot_confusion_matrix(ax, fig, t_cm)
  wandb.log({"t_test/t_cm": wandb.Image(plt)})
  plt.close() 
  # Plot Super Classes Confusion Matrix
  fig, ax = plt.subplots(figsize=(5,5))
  sns.heatmap(s1_s_cm, annot=True, fmt='.2f', xticklabels=s1_classes, yticklabels=s1_classes, cmap=plt.cm.Blues)
  wandb.log({"s_test/s1_s_cm": wandb.Image(plt)})
  plt.close()
  fig, ax = plt.subplots(figsize=(5,5))
  sns.heatmap(s1_t_cm, annot=True, fmt='.2f', xticklabels=s1_classes, yticklabels=s1_classes, cmap=plt.cm.Blues)
  wandb.log({"t_test/s1_t_cm": wandb.Image(plt)})
  plt.close() 
  # Super Classes 2
  fig, ax = plt.subplots(figsize=(10,10))
  sns.heatmap(s2_s_cm, annot=True, fmt='.2f', xticklabels=s2_classes, yticklabels=s2_classes, cmap=plt.cm.Blues)
  wandb.log({"s_test/s2_s_cm": wandb.Image(plt)})
  plt.close()
  fig, ax = plt.subplots(figsize=(10,10))
  sns.heatmap(s2_t_cm, annot=True, fmt='.2f', xticklabels=s2_classes, yticklabels=s2_classes, cmap=plt.cm.Blues)
  wandb.log({"t_test/s2_t_cm": wandb.Image(plt)})
  plt.close() 
  # Savings 
  torch.save({
    'epoch': epoch,
    'args': args,
    's_cm': s_cm,
    't_cm': t_cm,
    's1_s_cm': s1_s_cm,
    's1_t_cm': s1_t_cm,
    's2_s_cm': s2_s_cm,
    's2_t_cm': s2_t_cm,
    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    },'/storage/TEV/sabbes/model_weights_{}.pth'.format(args.exp))
  # Savings on Wandb
  if hasattr(model, 'module'):
    model.module.save(os.path.join(wandb.run.dir, "model.{}".format(args.exp)))
  else:
    model.save(os.path.join(wandb.run.dir, "model.{}".format(args.exp)))
  np.save(os.path.join(wandb.run.dir, "s_cm_{}".format(args.exp)), s_cm)
  np.save(os.path.join(wandb.run.dir, "t_cm_{}".format(args.exp)), t_cm)
  np.save(os.path.join(wandb.run.dir, "s1_s_cm_{}".format(args.exp)), s1_s_cm)
  np.save(os.path.join(wandb.run.dir, "s1_t_cm_{}".format(args.exp)), s1_t_cm)
  np.save(os.path.join(wandb.run.dir, "s2_s_cm_{}".format(args.exp)), s2_s_cm)
  np.save(os.path.join(wandb.run.dir, "s2_t_cm_{}".format(args.exp)), s2_t_cm)
  wandb.finish()

def source_train_step(epoch, args, model, data_loader, optimizer, logger):
  nb_samples = 0
  cumulative_loss = 0.
  total_correct = 0
  batch_step = 0
  model.train() 
  n_total_steps = len(data_loader)
  for batch in data_loader:
    data, (labels, _ ,_)  = batch
    preds = model(data)
    loss = F.cross_entropy(preds,labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    nb_samples += data.shape[0]
    cumulative_loss += loss.item()
    total_correct += get_num_correct(preds, labels)   
    batch_step += 1
    if (batch_step + 1) % 200 == 0:
      logger.info('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'.format(epoch+1, args.n_epochs, batch_step+1,n_total_steps, loss.item()))    
  # compute average loss and accuracy
  average_loss = cumulative_loss / nb_samples
  average_accuracy = total_correct / nb_samples * 100
  return average_loss, average_accuracy 




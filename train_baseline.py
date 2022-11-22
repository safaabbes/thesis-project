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

from utils import *

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)

target_classes = ("airplane", "ambulance", "apple", "backpack", "banana", "bathtub", "bear", "bed", "bee", "bicycle", "bird", "book", "bridge", 
                "bus", "butterfly", "cake", "calculator", "camera", "car", "cat", "chair", "clock", "cow", "dog", "dolphin", "donut", "drums", 
                "duck", "elephant", "fence", "fork", "horse", "house", "rabbit", "scissors", "sheep", "strawberry", "table", "telephone", "truck")


def run_baseline_epochs( s_train_dl, s_test_dl, t_train_dl, t_test_dl, model, args, optimizer, scheduler, logger):  
  # Setting Wandb
  wandb.init(
      project='Source-Sentry-Testing', #Source-Only-ResNet50 OR Super-Class-Model-ResNet50
      name=args.exp,
      config = {
                "source": args.source,
                "target": args.target,
                "learning_rate": args.lr,
                "epochs": args.n_epochs,
                "batch_size": args.bs,
                "optimizer": args.optimizer,
                "scheduler": args.scheduler,
                "step": args.step,
                "w_decay": args.wd,
                "momentum": args.momentum,
                })

  #set starting time
  since = time.time()
  for epoch in range(args.n_epochs):
    # OPTION 1: Source-Only Training
    train_loss , train_accuracy = source_train_step(epoch, args, model,s_train_dl, optimizer, logger)
    # Testing
    s_test_accuracy, s_per_cls_avg_acc, s_cm= test_step(args, model,s_test_dl, logger)
    t_test_accuracy, t_per_cls_avg_acc, t_cm = test_step(args, model,t_test_dl, logger)
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
    # Scheduler Step
    scheduler.step()
    
  # Plot Confusion Matrix
  plt.figure(figsize=(50,50))
  plot_confusion_matrix(s_cm, target_classes)
  wandb.log({"Source/Confusion Matrix": plt})
  plt.close()
  plt.figure(figsize=(50,50))
  plot_confusion_matrix(t_cm, target_classes)
  wandb.log({"Target/Confusion Matrix": plt})
  plt.close() 
  # Log time
  duration = time.time() - since
  logger.info('Training duration: {}'.format(str(datetime.timedelta(seconds=duration))))
  # Save model
  torch.save(model.state_dict(), '/storage/TEV/sabbes/model_weights_{}.pth'.format(args.exp))
  model.save(os.path.join(wandb.run.dir, "model.{}".format(args.exp)))
  wandb.finish()
  
  

def source_train_step(epoch, args, model, data_loader, optimizer, logger):
  nb_samples = 0
  cumulative_loss = 0.
  total_correct = 0
  batch_step = 0
  # set the network to training mode: particularly important when using dropout!
  model.train() 
  # iterate over the training set
  n_total_steps = len(data_loader)
  for batch in data_loader:
    # logger.info('Train Batch {} out of {}'.format(batch_idx+1, len(data_loader)))
    data, (labels, _ )  = batch
    # forward pass
    preds = model(data)
    # loss computation
    loss = F.cross_entropy(preds,labels)
    # backward pass
    loss.backward()
    # parameters update
    optimizer.step()
    # gradients reset
    optimizer.zero_grad()
    # fetch prediction and loss value
    nb_samples += data.shape[0]
    cumulative_loss += loss.item()
    total_correct += get_num_correct(preds, labels)   
    # Logging update
    batch_step += 1
    if (batch_step + 1) % 100 == 0:
      logger.info('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'.format(epoch+1, args.n_epochs, batch_step+1,n_total_steps, loss.item()))
      
  # compute average loss and accuracy
  average_loss = cumulative_loss / nb_samples
  average_accuracy = total_correct / nb_samples * 100
    
  return average_loss, average_accuracy 




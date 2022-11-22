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


def run_model_epochs( s_train_dl, s_test_dl, t_train_dl, t_test_dl, model, args, optimizer, scheduler, logger):  
  # Setting Wandb
  wandb.init(
      project='New-Architecture-Testing', #Source-Only-ResNet50 OR Super-Class-Model-ResNet50
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
                "w_decay": args.w_decay,
                "momentum": args.momentum,
                })

  #set starting time
  since = time.time()
  for epoch in range(args.n_epochs):
    # Multi-Task using super-classes Training
    s_avg_acc, sb_avg_acc, tb_avg_acc = model_train_step(epoch, args, model,s_train_dl, t_train_dl, optimizer, logger)
    # Testing
    s_test_accuracy, s_per_cls_avg_acc, s_cm = model_test_step(args, model,s_test_dl, logger)
    t_test_accuracy, t_per_cls_avg_acc, t_cm = model_test_step(args, model,t_test_dl, logger)
    # Log Results
    logger.info('Epoch: {:d}'.format(epoch+1))
    logger.info('\t Source AVG ACC {:.2f}, Source TASK AVG ACC  {:.2f}, Target TASK AVG ACC  {:.2f}'.format(s_avg_acc, sb_avg_acc, tb_avg_acc))
    logger.info('\t Source Test accuracy {:.2f}, Source per_cls_avg_acc {:.2f}'.format(s_test_loss, s_test_accuracy, s_per_cls_avg_acc))
    logger.info('\t Target Test accuracy {:.2f}, Target per_cls_avg_acc {:.2f}'.format(t_test_loss, t_test_accuracy, t_per_cls_avg_acc))
    logger.info('-----------------------------------------------------------------------')
    # Log results to Wandb
    metrics = {#Train Results
               "train/main_avg_acc": s_avg_acc, 
               "train/sb_avg_acc": sb_avg_acc,
               "train/tb_avg_acc": tb_avg_acc,
               #Source Test Results
               "s_test/test_loss":s_test_loss,
               "s_test/test_per_cls_avg_acc":s_per_cls_avg_acc,
               "s_test/test_avg_acc":s_test_accuracy,
               #Target Test Results
               "t_test/test_loss":t_test_loss,
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
  
  
  
def model_train_step(epoch, args, model, source_dl, target_dl, optimizer, logger):
  nb_source_samples = 0
  nb_target_samples = 0
  sm_total_correct = 0
  sb_total_correct = 0
  tb_total_correct = 0
  alpha = 1 #weight of source task loss
  gamma = 1 #weight of target task loss
  batch_step = 0
  logger.info("Start Training")
  # set the network to training mode: particularly important when using dropout!
  model.train() 
  # iterate over the training set
  n_total_steps = min(len(source_dl), len(target_dl))
  for batch_idx in range(n_total_steps):
    # DOUBTS ABOUT THIS PART OF GOING THROUGH BOTH DL SIMULTANEOUSLY 
    s_data, (s_labels, s_super_labels)  = next(iter(source_dl))
    t_data, ( _ , t_super_labels)  = next(iter(target_dl))
    # forward pass
    main_preds = model(s_data, 'main')
    s_branch_preds = model(s_data, 'branch')
    t_branch_preds = model(t_data, 'branch')
    # loss computation
    main_loss = F.cross_entropy(main_preds, s_labels)
    s_branch_loss = F.cross_entropy(s_branch_preds, s_super_labels)
    t_branch_loss = F.cross_entropy(t_branch_preds, t_super_labels)
    total_loss = main_loss + alpha*s_branch_loss + gamma*t_branch_loss
    # backward pass
    total_loss.backward()
    # parameters update
    optimizer.step()
    # gradients reset
    optimizer.zero_grad()
    # fetch prediction and loss value
    nb_source_samples += s_data.shape[0]
    nb_target_samples += t_data.shape[0]
    sm_total_correct += get_num_correct(main_preds, s_labels)
    sb_total_correct += get_num_correct(s_branch_preds, s_super_labels)
    tb_total_correct += get_num_correct(t_branch_loss, t_super_labels)   
    
    # Logging update
    if (batch_idx + 1) % 100 == 0:
      logger.info('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'.format(epoch+1, args.n_epochs, batch_idx+1,n_total_steps, main_loss.item()))
    
  # compute average accuracies
  s_avg_acc = sm_total_correct / nb_source_samples * 100
  sb_avg_acc = sb_total_correct / nb_source_samples * 100
  tb_avg_acc = tb_total_correct / nb_target_samples * 100
  
  return s_avg_acc, sb_avg_acc, tb_avg_acc



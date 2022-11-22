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
torch.cuda.manual_seed(1234)

target_classes = ("airplane", "ambulance", "apple", "backpack", "banana", "bathtub", "bear", "bed", "bee", "bicycle", "bird", "book", "bridge", 
                "bus", "butterfly", "cake", "calculator", "camera", "car", "cat", "chair", "clock", "cow", "dog", "dolphin", "donut", "drums", 
                "duck", "elephant", "fence", "fork", "horse", "house", "rabbit", "scissors", "sheep", "strawberry", "table", "telephone", "truck")


def run_epochs( s_train_dl, s_test_dl, t_train_dl, t_test_dl, model, args, optimizer, scheduler, logger):  
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
    # OPTION 1: Source-Only Training
    # train_loss , train_accuracy = source_train_step(epoch, args, model,s_train_dl, optimizer, logger)
    # OPTION 2: Multi-Task using super-classes Training
    train_loss , train_accuracy = new_train_step(epoch, args, model,s_train_dl, t_train_dl, optimizer, logger)
    # Testing
    s_test_loss, s_test_accuracy, s_per_cls_avg_acc, s_acc_each_cls, s_per_cls_samples, s_cm= test_step(args, model,s_test_dl, logger)
    t_test_loss, t_test_accuracy, t_per_cls_avg_acc, t_acc_each_cls, t_per_cls_samples, t_cm = test_step(args, model,t_test_dl, logger)
    # Log Results
    logger.info('Epoch: {:d}'.format(epoch+1))
    logger.info('\t Source Train loss {:.5f}, Source Train accuracy {:.2f}'.format(train_loss, train_accuracy))
    logger.info('\t Source Test loss {:.5f}, Source Test accuracy {:.2f}, Source per_cls_avg_acc {:.2f}'.format(s_test_loss, s_test_accuracy, s_per_cls_avg_acc))
    logger.info('\t Target Test loss {:.5f}, Target Test accuracy {:.2f}, Target per_cls_avg_acc {:.2f}'.format(t_test_loss, t_test_accuracy, t_per_cls_avg_acc))
    logger.info('-----------------------------------------------------------------------')
    print('s_per_cls_samples: ', s_per_cls_samples)
    print('s_acc_each_cls: ', s_acc_each_cls)
    print('t_per_cls_samples: ', t_per_cls_samples)
    print('t_acc_each_cls: ', t_acc_each_cls)
    # Log results to Wandb
    metrics = {#Train Results
               "train/train_loss": train_loss, 
               "train/train_acc": train_accuracy,
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
  
def get_num_correct(preds, labels):
  return preds.argmax(dim=1).eq(labels).sum().item()

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
      cumulative_loss += loss.item() #.item() is needed to extract scalars from tensors
      total_correct += get_num_correct(preds, labels)
      # Save predictions
      all_preds = torch.cat((all_preds, preds.argmax(dim=1).int()), dim=0)
      all_true = torch.cat((all_true, labels.int()), dim=0)
  
    # compute average loss and accuracy
    average_loss = cumulative_loss / nb_samples
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
    per_cls_acc_list = { i: np.round(per_cls_acc_vec[i], 2) for i in range(len(per_cls_acc_vec))}
    per_cls_samples = { i: cm[i,:].sum() for i in range(len(target_classes))}

  return average_loss, average_accuracy, per_cls_avg_acc, per_cls_acc_list, per_cls_samples, cm


def new_train_step(epoch, args, model, source_dl, target_dl, optimizer, logger):
  nb_samples = 0
  cumulative_loss = 0.
  total_correct = 0
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
    nb_samples += s_data.shape[0]
    cumulative_loss += main_loss.item()
    total_correct += get_num_correct(main_preds, s_labels)   
    # Logging update
    if (batch_idx + 1) % 100 == 0:
      logger.info('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'.format(epoch+1, args.n_epochs, batch_idx+1,n_total_steps, main_loss.item()))
    
  # compute average loss and accuracy
  average_loss = cumulative_loss / nb_samples
  average_accuracy = total_correct / nb_samples * 100
  
  return average_loss, average_accuracy





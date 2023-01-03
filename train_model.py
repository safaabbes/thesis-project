import os
import sys
import torch
import time
import datetime
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from itertools import cycle
import wandb
import seaborn as sns

from utils import *
from classes import *

def train_model( s_train_dl, s_test_dl, t_train_dl, t_test_dl, model, args, optimizer, logger):  
  # Setting Wandb
  wandb.init(
      project='New_B4_{}'.format(args.task), 
      name=args.exp,
      config = {
                "source": args.source,
                "target": args.target,
                "epochs": args.n_epochs,
                "batch_size": args.bs,
                "balance": args.cbt,
                # New Hyper-parameters
                "mu1": args.mu1,
                "mu2": args.mu2,
                "mu3": args.mu3,
                })
  #set starting time
  since = time.time()
  # wandb.watch(model)
  for epoch in range(args.n_epochs):
    if args.task == 'run_model_v1':
      s_avg_acc, sb_avg_acc, tb_avg_acc, all_losses  = model_v1_train_step(epoch, args, model,s_train_dl, t_train_dl, optimizer, logger)
    elif args.task == 'run_model_v2':
      s_avg_acc, sb_avg_acc, tb_avg_acc, all_losses  = model_v2_train_step(epoch, args, model,s_train_dl, t_train_dl, optimizer, logger)
    elif args.task == 'run_model_v3':
      s_avg_acc, sb_avg_acc, tb_avg_acc, all_losses = model_v3_train_step(epoch, args, model,s_train_dl, t_train_dl, optimizer, logger)
    # Plot Samples Per Cls
    # if epoch == 0:
    #   plot_samples_per_cls(instances_list)
    # Testing  
    s_test_loss, s_test_accuracy, s_per_cls_avg_acc, s_cm, s1_s_cm, s2_s_cm = test_step(args, model,s_test_dl, logger)
    t_test_loss, t_test_accuracy, t_per_cls_avg_acc, t_cm, s1_t_cm, s2_t_cm= test_step(args, model,t_test_dl, logger)
    # Log Results
    logger.info('Epoch: {:d}'.format(epoch+1))
    logger.info('\t Source AVG ACC {:.2f}, Source TASK AVG ACC  {:.2f}, Target TASK AVG ACC  {:.2f}'.format(s_avg_acc, sb_avg_acc, tb_avg_acc))
    logger.info('\t Source Test accuracy {:.2f}, Source per_cls_avg_acc {:.2f}'.format(s_test_accuracy, s_per_cls_avg_acc))
    logger.info('\t Target Test accuracy {:.2f}, Target per_cls_avg_acc {:.2f}'.format(t_test_accuracy, t_per_cls_avg_acc))
    logger.info('-----------------------------------------------------------------------')
    # Log results to Wandb 
    wandb.log({
        "epoch": epoch,
        # Training Losses
        "main_loss": all_losses[0],
        "sb_loss": all_losses[1],
        "tb_loss": all_losses[2],
        "total_loss": all_losses[3],
        # Train Accuracies
        "main_avg_acc": s_avg_acc, 
        "sb_avg_acc": sb_avg_acc,
        "tb_avg_acc": tb_avg_acc,
        # Source Test Results
        "s_test/s_test_loss": s_test_loss,
        "s_test/test_per_cls_avg_acc":s_per_cls_avg_acc,
        "s_test/test_avg_acc":s_test_accuracy,
        # Target Test Results
        "t_test/t_test_loss": t_test_loss,
        "t_test/test_per_cls_avg_acc":t_per_cls_avg_acc,
        "t_test/test_avg_acc":t_test_accuracy,
        })
    
    if (epoch) % args.freq_saving == 0:
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
      fig, ax = plt.subplots(figsize=(15,15))
      sns.heatmap(s2_s_cm, annot=True, fmt='.2f', xticklabels=s2_classes, yticklabels=s2_classes, cmap=plt.cm.Blues)
      wandb.log({"s_test/s2_s_cm": wandb.Image(plt)})
      plt.close()
      fig, ax = plt.subplots(figsize=(15,15))
      sns.heatmap(s2_t_cm, annot=True, fmt='.2f', xticklabels=s2_classes, yticklabels=s2_classes, cmap=plt.cm.Blues)
      wandb.log({"t_test/s2_t_cm": wandb.Image(plt)})
      plt.close() 
      
  # Log time
  duration = time.time() - since
  logger.info('Training duration: {}'.format(str(datetime.timedelta(seconds=duration))))
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
  
def model_v1_train_step(epoch, args, model, source_dl, target_dl, optimizer, logger):
  nb_source_samples = 0
  nb_target_samples = 0
  sm_total_correct = 0
  sb_total_correct = 0
  tb_total_correct = 0
  cml_main_loss = 0.
  cml_s_branch_loss = 0.
  cml_t_branch_loss = 0.
  cml_total_loss = 0.
  # instances_list = [0] * 40
  logger.info("Start Training")
  model.train() 
  n_total_steps = max(len(source_dl), len(target_dl))
  for batch_idx, ((s_data, (s_labels, s_super_labels,_)), (t_data, ( _ , t_super_labels,_))) in enumerate(zip(cycle(source_dl), cycle(target_dl))):
    # Note: To avoid Cuda Out of Memory for Real: Remove cycle for source_dl
    if batch_idx == n_total_steps:
      break
    # forward pass
    main_preds = model(s_data)
    s_branch_preds = model(s_data, 'b1')
    t_branch_preds = model(t_data, 'b1')
    # loss computation
    main_loss = F.cross_entropy(main_preds, s_labels)
    s_branch_loss = F.cross_entropy(s_branch_preds, s_super_labels)
    t_branch_loss = F.cross_entropy(t_branch_preds, t_super_labels)
    total_loss = args.mu1 * main_loss + args.mu2 * s_branch_loss + args.mu3 * t_branch_loss
    # backward pass
    total_loss.backward()
    # parameters update
    optimizer.step()
    # gradients reset
    optimizer.zero_grad()
    # fetch prediction and loss value
    nb_source_samples += s_data.shape[0]
    nb_target_samples += t_data.shape[0]
    cml_main_loss += main_loss.item()
    cml_s_branch_loss += s_branch_loss.item()
    cml_t_branch_loss += t_branch_loss.item()
    cml_total_loss += total_loss.item()
    sm_total_correct += get_num_correct(main_preds, s_labels)
    sb_total_correct += get_num_correct(s_branch_preds, s_super_labels)
    tb_total_correct += get_num_correct(t_branch_preds, t_super_labels)     
    # for label in s_labels:
    #   instances_list[label] += 1      
    # Logging update
    if (batch_idx + 1) % 200 == 0:
      logger.info('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'.format(epoch+1, args.n_epochs, batch_idx+1,n_total_steps, main_loss.item()))
      
  # Plot number of samples loaded 
  logger.info('Number of Source Samples used {} / {}'.format(nb_source_samples , len(source_dl.dataset)))
  logger.info('Number of Target Samples used {} / {}'.format(nb_target_samples , len(target_dl.dataset)))
  # compute average losses
  avg_main_loss = cml_main_loss / nb_source_samples
  avg_s_branch_loss = cml_s_branch_loss / nb_source_samples
  avg_t_branch_loss = cml_t_branch_loss / nb_source_samples
  avg_total_loss = cml_total_loss / nb_source_samples
  all_losses = [avg_main_loss, avg_s_branch_loss, avg_t_branch_loss, avg_total_loss]
  # compute average accuracies
  s_avg_acc = sm_total_correct / nb_source_samples * 100
  sb_avg_acc = sb_total_correct / nb_source_samples * 100
  tb_avg_acc = tb_total_correct / nb_target_samples * 100
  
  return s_avg_acc, sb_avg_acc, tb_avg_acc, all_losses


def model_v2_train_step(epoch, args, model, source_dl, target_dl, optimizer, logger):
  nb_source_samples = 0
  nb_target_samples = 0
  sm_total_correct = 0
  sb_total_correct = 0
  tb_total_correct = 0
  cml_main_loss = 0.
  cml_s_branch_loss = 0.
  cml_t_branch_loss = 0.
  cml_total_loss = 0.
  logger.info("Start Training")
  model.train() 
  n_total_steps = max(len(source_dl), len(target_dl))
  for batch_idx, ((s_data, (s_labels,_,s_super_labels)), (t_data, ( _ ,_,t_super_labels))) in enumerate(zip(cycle(source_dl), cycle(target_dl))):
    # Note: To avoid Cuda Out of Memory for Real: Remove cycle for source_dl
    if batch_idx == n_total_steps:
      break
    # forward pass
    main_preds = model(s_data, 'main')
    s_branch_preds = model(s_data, 'b2')
    t_branch_preds = model(t_data, 'b2')
    # loss computation
    main_loss = F.cross_entropy(main_preds, s_labels)
    s_branch_loss = F.cross_entropy(s_branch_preds, s_super_labels)
    t_branch_loss = F.cross_entropy(t_branch_preds, t_super_labels)
    total_loss = args.mu1 * main_loss + args.mu2 * s_branch_loss + args.mu3 * t_branch_loss
    # backward pass
    total_loss.backward()
    # parameters update
    optimizer.step()
    # gradients reset
    optimizer.zero_grad()
    # fetch prediction and loss value
    nb_source_samples += s_data.shape[0]
    nb_target_samples += t_data.shape[0]
    cml_main_loss += main_loss.item()
    cml_s_branch_loss += s_branch_loss.item()
    cml_t_branch_loss += t_branch_loss.item()
    cml_total_loss += total_loss.item()
    sm_total_correct += get_num_correct(main_preds, s_labels)
    sb_total_correct += get_num_correct(s_branch_preds, s_super_labels)
    tb_total_correct += get_num_correct(t_branch_preds, t_super_labels)     
    # Logging update
    if (batch_idx + 1) % 200 == 0:
      logger.info('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'.format(epoch+1, args.n_epochs, batch_idx+1,n_total_steps, main_loss.item()))
      
  # Plot number of samples loaded 
  logger.info('Number of Source Samples used {} / {}'.format(nb_source_samples , len(source_dl.dataset)))
  logger.info('Number of Target Samples used {} / {}'.format(nb_target_samples , len(target_dl.dataset)))
  # compute average losses
  avg_main_loss = cml_main_loss / nb_source_samples
  avg_s_branch_loss = cml_s_branch_loss / nb_source_samples
  avg_t_branch_loss = cml_t_branch_loss / nb_source_samples
  avg_total_loss = cml_total_loss / nb_source_samples
  all_losses = [avg_main_loss, avg_s_branch_loss, avg_t_branch_loss, avg_total_loss]
  # compute average accuracies
  s_avg_acc = sm_total_correct / nb_source_samples * 100
  sb_avg_acc = sb_total_correct / nb_source_samples * 100
  tb_avg_acc = tb_total_correct / nb_target_samples * 100
  
  return s_avg_acc, sb_avg_acc, tb_avg_acc, all_losses

def model_v3_train_step(epoch, args, model, source_dl, target_dl, optimizer, logger):
  nb_source_samples = 0
  nb_target_samples = 0
  sm_total_correct = 0
  sb1_total_correct = 0
  tb1_total_correct = 0
  sb2_total_correct = 0
  tb2_total_correct = 0
  cml_main_loss = 0.
  cml_s1_branch_loss = 0.
  cml_t1_branch_loss = 0.
  cml_s2_branch_loss = 0.
  cml_t2_branch_loss = 0.
  cml_total_loss = 0.
  # instances_list = [0] * 40
  logger.info("Start Training")
  model.train() 
  n_total_steps = max(len(source_dl), len(target_dl))
  for batch_idx, ((s_data, (s_labels,s1_super_labels,s2_super_labels)), (t_data, ( _ , t1_super_labels,t2_super_labels))) in enumerate(zip(cycle(source_dl), cycle(target_dl))):
    if batch_idx == n_total_steps:
      break
    # forward pass
    main_preds = model(s_data, 'main')
    s1_branch_preds = model(s_data, 'b1')
    t1_branch_preds = model(t_data, 'b1')
    s2_branch_preds = model(s_data, 'b2')
    t2_branch_preds = model(t_data, 'b2')
    # loss computation
    main_loss = F.cross_entropy(main_preds, s_labels)
    s1_branch_loss = F.cross_entropy(s1_branch_preds, s1_super_labels)
    t1_branch_loss = F.cross_entropy(t1_branch_preds, t1_super_labels)
    s2_branch_loss = F.cross_entropy(s2_branch_preds, s2_super_labels)
    t2_branch_loss = F.cross_entropy(t2_branch_preds, t2_super_labels)
    total_loss = args.mu1 * main_loss + args.mu2 * s1_branch_loss + args.mu3 * t1_branch_loss + args.mu2 * s2_branch_loss + args.mu3 * t2_branch_loss
    # backward pass
    total_loss.backward()
    # parameters update
    optimizer.step()
    # gradients reset
    optimizer.zero_grad()
    # fetch prediction and loss value
    nb_source_samples += s_data.shape[0]
    nb_target_samples += t_data.shape[0]
    cml_main_loss += main_loss.item()
    cml_s1_branch_loss += s1_branch_loss.item()
    cml_t1_branch_loss += t1_branch_loss.item()
    cml_s2_branch_loss += s2_branch_loss.item()
    cml_t2_branch_loss += t2_branch_loss.item()
    cml_total_loss += total_loss.item()
    sm_total_correct += get_num_correct(main_preds, s_labels)
    sb1_total_correct += get_num_correct(s1_branch_preds, s1_super_labels)
    tb1_total_correct += get_num_correct(t1_branch_preds, t1_super_labels)     
    sb2_total_correct += get_num_correct(s2_branch_preds, s2_super_labels)
    tb2_total_correct += get_num_correct(t2_branch_preds, t2_super_labels)    
    # for label in s_labels:
    #   instances_list[label] += 1      
    # Logging update
    if (batch_idx + 1) % 200 == 0:
      logger.info('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'.format(epoch+1, args.n_epochs, batch_idx+1,n_total_steps, main_loss.item()))
      
  # Plot number of samples loaded 
  logger.info('Number of Source Samples used {} / {}'.format(nb_source_samples , len(source_dl.dataset)))
  logger.info('Number of Target Samples used {} / {}'.format(nb_target_samples , len(target_dl.dataset)))
  # compute average losses
  avg_main_loss = cml_main_loss / nb_source_samples
  avg_s1_branch_loss = cml_s1_branch_loss / nb_source_samples
  avg_t1_branch_loss = cml_t1_branch_loss / nb_source_samples
  avg_s2_branch_loss = cml_s2_branch_loss / nb_source_samples
  avg_t2_branch_loss = cml_t2_branch_loss / nb_source_samples
  avg_total_loss = cml_total_loss / nb_source_samples
  all_losses = [avg_main_loss, avg_s1_branch_loss, avg_t1_branch_loss, avg_total_loss]
  # compute average accuracies
  s_avg_acc = sm_total_correct / nb_source_samples * 100
  sb1_avg_acc = sb1_total_correct / nb_source_samples * 100
  tb1_avg_acc = tb1_total_correct / nb_target_samples * 100
  sb2_avg_acc = sb2_total_correct / nb_source_samples * 100
  tb2_avg_acc = tb2_total_correct / nb_target_samples * 100
  
  return s_avg_acc, sb1_avg_acc, tb1_avg_acc, all_losses

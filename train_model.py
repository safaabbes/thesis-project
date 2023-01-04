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

def train_model( s_train_dl, s_test_dl, t_train_dl, t_test_dl, model, args, optimizer, criterion, logger):  
  # Setting Wandb
  wandb.init(
      project='EXP0', 
      name=args.exp,
      config = {"model_type": args.model_type,
                "source": args.source,
                "target": args.target,
                "epochs": args.n_epochs,
                "batch_size": args.bs,
                "balance": args.balance_mini_batches,
                "lr": args.lr,
                "reduction": args.reduction,
                "mean_bs": args.mean_bs,
                "mu1": args.mu1,
                "mu2": args.mu2,
                "mu3": args.mu3,
                })
  #set starting time
  since = time.time()
  # wandb.watch(model)
  for epoch in range(args.n_epochs):
    if args.model_type == 'R50_2H':
      train_stats = train_step_2H(epoch, args, model,s_train_dl, t_train_dl, optimizer, criterion, logger)
    elif args.model_type == 'R50_1H':
      train_stats = train_step_1H(epoch, args, model,s_train_dl, t_train_dl, optimizer, criterion, logger)
    # Testing  
    s_test_stats = test_step(args, model,s_test_dl, criterion, logger)
    t_test_stats = test_step(args, model,t_test_dl, criterion, logger)
    
    # Log Results
    # if (epoch) % args.freq_saving == 0:
    logger.info('Epoch: {:d}'.format(epoch+1)) 
    logger.info('\t Source Test Accuracies')
    logger.info('\t S_OA1: {:.2f}, S_MCA1: {:.2f}, S_OA2: {:.2f}, S_MCA2: {:.2f}'.format(
        s_test_stats['oa1'], s_test_stats['mca1'], s_test_stats['oa2'], s_test_stats['mca2']))
    logger.info('\t Target Test Accuracies')
    logger.info('\t T_OA1: {:.2f}, T_MCA1: {:.2f}, T_OA2: {:.2f}, T_MCA2: {:.2f}'.format(
        t_test_stats['oa1'], t_test_stats['mca1'], t_test_stats['oa2'], t_test_stats['mca2']))
    logger.info('-----------------------------------------------------------------------')
    
    # Log results to Wandb 
    wandb.log({
        "epoch": epoch,
        # Train Accuracies
        "train_oa1": train_stats['oa1'], 
        "train_mca1": train_stats['mca1'], 
        "train_oa2": train_stats['oa2'], 
        "train_mca2": train_stats['mca2'], 
        # Source Test Results
        "s_test/oa1": s_test_stats['oa1'],
        "s_test/oa2": s_test_stats['oa2'],
        "s_test/mca1": s_test_stats['mca1'],
        "s_test/mca2": s_test_stats['mca2'],
        # Target Test Results
        "t_test/oa1": t_test_stats['oa1'],
        "t_test/oa2": t_test_stats['oa2'],
        "t_test/mca1": t_test_stats['mca1'],
        "t_test/mca2": t_test_stats['mca2'],
        })
  # Log time
  duration = time.time() - since
  logger.info('Training duration: {}'.format(str(datetime.timedelta(seconds=duration))))
  # Savings 
  torch.save({
    'epoch': epoch,
    'args': args,
    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    },'/storage/TEV/sabbes/model_weights_{}.pth'.format(args.exp))
  # Savings on Wandb
  if hasattr(model, 'module'):
    model.module.save(os.path.join(wandb.run.dir, "model.{}".format(args.exp)))
  else:
    model.save(os.path.join(wandb.run.dir, "model.{}".format(args.exp)))
  wandb.finish()
  

def train_step_2H(epoch, args, model, source_dl, target_dl, optimizer, criterion, logger):
  nb_source_samples, nb_target_samples = 0,0
  running_loss, running_loss1_source, running_loss2_source, running_loss2_target = list(), list(), list(), list()
  running_oa1, running_mca1_num, running_mca1_den = list(), list(), list()
  running_oa2, running_mca2_num, running_mca2_den = list(), list(), list()
  logger.info("Start Training")
  model.train() 
  n_total_steps = max(len(source_dl), len(target_dl))
  
  # Loop over the bigger dataloader
  # for batch_idx, (data_source, data_target) in enumerate(zip(cycle(source_dl), cycle(target_dl))):
  #   if batch_idx == n_total_steps:
  #     break
   
  # Loop on target dataloader
  dataloader_iterator = iter(source_dl)
  for data_target in target_dl:
    try:
        data_source = next(dataloader_iterator)
    except StopIteration:
        dataloader_iterator = iter(source_dl)
        data_source = next(dataloader_iterator)

    # Load source mini-batch
    images_source, (s_labels, s1_super_labels, s2_super_labels) = data_source
    
    # Load target mini-batch
    images_target, (t_labels, t1_super_labels, t2_super_labels) = data_target
    
    # Zero the parameters gradients
    optimizer.zero_grad()
    
    # Forward pass for source data
    logits1_source, logits2_source = model(images_source)
    _, preds1_source = torch.max(logits1_source, dim=1)
    _, preds2_source = torch.max(logits2_source, dim=1)

    # Forward pass for target data
    logits1_target, logits2_target = model(images_target)
    _, preds1_target = torch.max(logits1_target, dim=1)
    _, preds2_target = torch.max(logits2_target, dim=1)

    # Losses
    main_loss = args.mu1 * criterion(logits1_source, s_labels)
    s_branch_loss = args.mu2 * criterion(logits2_source, s2_super_labels)
    t_branch_loss = args.mu3 * criterion(logits2_target, t2_super_labels)
    loss = main_loss + s_branch_loss + t_branch_loss
    if args.mean_bs == True:
      loss = loss * args.bs

    # Back-propagation
    loss.backward()

    # Optimizer step
    optimizer.step()

    # fetch prediction and loss value
    nb_source_samples += images_source.shape[0]
    nb_target_samples += images_target.shape[0]
    
    # Update Losses
    running_loss.append(loss.item())
    running_loss1_source.append(main_loss.item())
    running_loss2_source.append(s_branch_loss.item())
    running_loss2_target.append(t_branch_loss.item())
    
    # Update metrics
    oa1 = torch.sum(preds1_source == s_labels.squeeze()) / len(s_labels)
    running_oa1.append(oa1.item())
    mca1_num = torch.sum(
        torch.nn.functional.one_hot(preds1_source, num_classes=40) * \
        torch.nn.functional.one_hot(s_labels, num_classes=40), dim=0)
    mca1_den = torch.sum(
        torch.nn.functional.one_hot(s_labels, num_classes=40), dim=0)
    running_mca1_num.append(mca1_num.detach().cpu().numpy())
    running_mca1_den.append(mca1_den.detach().cpu().numpy())

    oa2 = torch.sum(preds2_source == s2_super_labels.squeeze()) / len(s2_super_labels)
    running_oa2.append(oa2.item())
    mca2_num = torch.sum(
        torch.nn.functional.one_hot(preds2_source, num_classes=13) * \
        torch.nn.functional.one_hot(s2_super_labels, num_classes=13), dim=0)
    mca2_den = torch.sum(
        torch.nn.functional.one_hot(s2_super_labels, num_classes=13), dim=0)
    running_mca2_num.append(mca2_num.detach().cpu().numpy())
    running_mca2_den.append(mca2_den.detach().cpu().numpy())
        
    # # Logging update
    # if (batch_idx + 1) % 200 == 0:
    #   logger.info('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'.format(epoch+1, args.n_epochs, batch_idx+1, n_total_steps, loss.item()))
      
  # Plot number of samples loaded 
  logger.info('Number of Source Samples used {} / {}'.format(nb_source_samples , len(source_dl.dataset)))
  logger.info('Number of Target Samples used {} / {}'.format(nb_target_samples , len(target_dl.dataset)))
  
  # Update MCA metric
  mca1_num = np.sum(running_mca1_num, axis=0)
  mca1_den = 1e-16 + np.sum(running_mca1_den, axis=0)
  mca2_num = np.sum(running_mca2_num, axis=0)
  mca2_den = 1e-16 + np.sum(running_mca2_den, axis=0)

  stats = {
    'loss': np.mean(running_loss),
    'loss1_source': np.mean(running_loss1_source),
    'loss2_source': np.mean(running_loss2_source),
    'loss2_target': np.mean(running_loss2_target),
    'oa1': np.mean(running_oa1)*100,
    'mca1': np.mean(mca1_num/mca1_den)*100,
    'oa2': np.mean(running_oa2)*100,
    'mca2': np.mean(mca2_num/mca2_den)*100
      }
  
  return stats


def train_step_1H(epoch, args, model, source_dl, target_dl, optimizer, criterion, logger):
  nb_source_samples, nb_target_samples = 0,0
  running_loss, running_loss1_source, running_loss2_source, running_loss2_target = list(), list(), list(), list()
  running_oa1, running_mca1_num, running_mca1_den = list(), list(), list()
  running_oa2, running_mca2_num, running_mca2_den = list(), list(), list()
  logger.info("Start Training")
  model.train() 
  n_total_steps = max(len(source_dl), len(target_dl))
  
  # # Loop over the bigger dataloader
  # for batch_idx, (data_source, data_target) in enumerate(zip(cycle(source_dl), cycle(target_dl))):
  #   if batch_idx == n_total_steps:
  #     break
    
  # Loop on target dataloader
  dataloader_iterator = iter(source_dl)
  for data_target in target_dl:
    try:
        data_source = next(dataloader_iterator)
    except StopIteration:
        dataloader_iterator = iter(source_dl)
        data_source = next(dataloader_iterator)

    # Load source mini-batch
    images_source, (s_labels, s1_super_labels, s2_super_labels) = data_source
    
    # Load target mini-batch
    images_target, (t_labels, t1_super_labels, t2_super_labels) = data_target
    
    # Zero the parameters gradients
    optimizer.zero_grad()
        
    # Forward pass for source data
    logits1_source = model(images_source)
    _, preds1_source = torch.max(logits1_source, dim=1)

    tmp = np.load('mapping.npz')
    mapping = torch.tensor(tmp['data'], dtype=torch.float32, device=args.device, requires_grad=False)
    logits2_source = torch.mm(logits1_source, mapping) / (1e-6 + torch.sum(mapping, dim=0))
    _, preds2_source = torch.max(logits2_source, dim=1)

    # Forward pass for target data
    logits1_target = model(images_target)
    _, preds1_target = torch.max(logits1_target, dim=1)

    tmp = np.load('mapping.npz')
    mapping = torch.tensor(tmp['data'], dtype=torch.float32, device=args.device, requires_grad=False)
    logits2_target = torch.mm(logits1_target, mapping) / (1e-6 + torch.sum(mapping, dim=0))
    _, preds2_target = torch.max(logits2_target, dim=1)
    
    # Losses
    main_loss = args.mu1 * criterion(logits1_source, s_labels)
    s_branch_loss = args.mu2 * criterion(logits2_source, s2_super_labels)
    t_branch_loss = args.mu3 * criterion(logits2_target, t2_super_labels)
    loss = main_loss + s_branch_loss + t_branch_loss
    if args.mean_bs == True:
      loss = loss * args.bs
      
    # Back-propagation
    loss.backward()

    # Optimizer step
    optimizer.step()

    # fetch prediction and loss value
    nb_source_samples += images_source.shape[0]
    nb_target_samples += images_target.shape[0]
    
    # Update Losses
    running_loss.append(loss.item())
    running_loss1_source.append(main_loss.item())
    running_loss2_source.append(s_branch_loss.item())
    running_loss2_target.append(t_branch_loss.item())
    
    # Update metrics
    oa1 = torch.sum(preds1_source == s_labels.squeeze()) / len(s_labels)
    running_oa1.append(oa1.item())
    mca1_num = torch.sum(
        torch.nn.functional.one_hot(preds1_source, num_classes=40) * \
        torch.nn.functional.one_hot(s_labels, num_classes=40), dim=0)
    mca1_den = torch.sum(
        torch.nn.functional.one_hot(s_labels, num_classes=40), dim=0)
    running_mca1_num.append(mca1_num.detach().cpu().numpy())
    running_mca1_den.append(mca1_den.detach().cpu().numpy())

    oa2 = torch.sum(preds2_source == s2_super_labels.squeeze()) / len(s2_super_labels)
    running_oa2.append(oa2.item())
    mca2_num = torch.sum(
        torch.nn.functional.one_hot(preds2_source, num_classes=13) * \
        torch.nn.functional.one_hot(s2_super_labels, num_classes=13), dim=0)
    mca2_den = torch.sum(
        torch.nn.functional.one_hot(s2_super_labels, num_classes=13), dim=0)
    running_mca2_num.append(mca2_num.detach().cpu().numpy())
    running_mca2_den.append(mca2_den.detach().cpu().numpy())
        
    # Logging update
    # if (batch_idx + 1) % 200 == 0:
    #   logger.info('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'.format(epoch+1, args.n_epochs, batch_idx+1, n_total_steps, loss.item()))
      
  # Plot number of samples loaded 
  logger.info('Number of Source Samples used {} / {}'.format(nb_source_samples , len(source_dl.dataset)))
  logger.info('Number of Target Samples used {} / {}'.format(nb_target_samples , len(target_dl.dataset)))
  
  # Update MCA metric
  mca1_num = np.sum(running_mca1_num, axis=0)
  mca1_den = 1e-16 + np.sum(running_mca1_den, axis=0)
  mca2_num = np.sum(running_mca2_num, axis=0)
  mca2_den = 1e-16 + np.sum(running_mca2_den, axis=0)

  stats = {
    'loss': np.mean(running_loss),
    'loss1_source': np.mean(running_loss1_source),
    'loss2_source': np.mean(running_loss2_source),
    'loss2_target': np.mean(running_loss2_target),
    'oa1': np.mean(running_oa1)*100,
    'mca1': np.mean(mca1_num/mca1_den)*100,
    'oa2': np.mean(running_oa2)*100,
    'mca2': np.mean(mca2_num/mca2_den)*100
      }
  
  return stats


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
    # if (batch_idx + 1) % 200 == 0:
    #   logger.info('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'.format(epoch+1, args.n_epochs, batch_idx+1,n_total_steps, main_loss.item()))
      
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

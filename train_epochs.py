import os
import random
import torch
import time
import datetime
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import wandb
from sklearn.metrics import confusion_matrix


random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

target_classes = ["airplane", "ambulance", "apple", "backpack", "banana", "bathtub", "bear", "bed", "bee", "bicycle", "bird", "book", "bridge", 
                "bus", "butterfly", "cake", "calculator", "camera", "car", "cat", "chair", "clock", "cow", "dog", "dolphin", "donut", "drums", 
                "duck", "elephant", "fence", "fork", "horse", "house", "rabbit", "scissors", "sheep", "strawberry", "table", "telephone", "truck"]




def run_epochs(s_train_dl, s_test_dl, t_train_dl, t_test_dl, model, args, optimizer, scheduler, logger):  
  # Setting Wandb
  wandb.init(
      project='testing-original-resnet50-source',
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
    train_loss , train_accuracy = train_step(epoch, args, model,s_train_dl, optimizer, logger)
    s_test_loss, s_test_accuracy, s_per_cls_avg_acc, s_acc_each_cls = test_step(args, model,s_test_dl, logger)
    t_test_loss, t_test_accuracy, t_per_cls_avg_acc, t_acc_each_cls = test_step(args, model,t_test_dl, logger)

    # Log Results
    logger.info('Epoch: {:d}'.format(epoch+1))
    logger.info('\t Source Train loss {:.5f}, Source Train accuracy {:.2f}'.format(train_loss, train_accuracy))
    logger.info('\t Source Test loss {:.5f}, Source Test accuracy {:.2f}, Source per_cls_avg_acc {:.2f}'.format(s_test_loss, s_test_accuracy, s_per_cls_avg_acc))
    logger.info('Source acc_each_cls {:.5f}'.format(s_acc_each_cls))
    logger.info('\t Target Test loss {:.5f}, Target Test accuracy {:.2f}, Target per_cls_avg_acc {:.2f}'.format(t_test_loss, t_test_accuracy, t_per_cls_avg_acc))
    logger.info('Taget acc_each_cls {:.5f}'.format(t_acc_each_cls))
    logger.info('-----------------------------------------------------------------------')
  
    # Log results to Wandb
    metrics = {"train/train_loss": train_loss, 
               "train/train_acc": train_accuracy,
               "s_test/test_loss":s_test_loss,
               "s_test/test_acc":s_test_accuracy,
               "t_test/test_loss":t_test_loss,
               "t_test/test_acc":t_test_accuracy,
            }
    wandb.log({**metrics})

    # Scheduler Step
    scheduler.step()
  # Log time
  duration = time.time() - since
  logger.info('Training duration: {}'.format(str(datetime.timedelta(seconds=duration))))
  wandb.finish()
  # Save model
  model.save('/storage/TEV/sabbes/')
  model.save(os.path.join(wandb.run.dir, "model.{}".format(args.exp)))
  


def train_step(epoch, args, model, data_loader, optimizer, logger):
  samples = 0.
  cumulative_loss = 0.
  cumulative_accuracy = 0.
  # set the network to training mode: particularly important when using dropout!
  model.train() 
  
  # iterate over the training set
  n_total_steps = len(data_loader)
  for batch_idx in range(len(data_loader)):
    # logger.info('Train Batch {} out of {}'.format(batch_idx+1, len(data_loader)))
    data, (labels, super_labels)  = next(iter(data_loader))
    # forward pass
    outputs = model(data)
    # loss computation
    loss = F.cross_entropy(outputs,labels)
    # backward pass
    loss.backward()
    # parameters update
    optimizer.step()
    # gradients reset
    optimizer.zero_grad()
    # fetch prediction and loss value
    samples += data.shape[0]
    cumulative_loss += loss.item()
    _, predicted = outputs.max(dim=1) # max() returns (maximum_value, index_of_maximum_value)
    # compute training accuracy
    cumulative_accuracy += predicted.eq(labels).sum().item()    
    # Logging update
    if (batch_idx + 1) % 100 == 0:
      logger.info('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'.format(epoch+1, args.n_epochs, batch_idx+1,n_total_steps, loss.item()))
    
  return cumulative_loss/samples, cumulative_accuracy/samples*100



def test_step(args, model, data_loader, logger):
  samples = 0.
  cumulative_loss = 0.
  cumulative_accuracy = 0.
  # set the network to evaluation mode
  model.eval() 

  # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
  start_test = True
  with torch.no_grad():
    # iterate over the test set
    for batch_idx in range(len(data_loader)):
      # logger.info('Test Batch {} out of {}'.format(batch_idx+1, len(data_loader)))
      data, (labels, super_labels)  = next(iter(data_loader))
      # forward pass
      outputs = model(data)
      # loss computation
      loss = F.cross_entropy(outputs, labels)
      # fetch prediction and loss value
      samples+=data.shape[0]
      cumulative_loss += loss.item() # Note: the .item() is needed to extract scalars from tensors
      
      if start_test:
        all_output = outputs.float().cpu()
        all_label = labels.float()
        start_test = False
      else:
        all_output = torch.cat((all_output, outputs.float().cpu()), 0)
        all_label = torch.cat((all_label, labels.float()), 0)
          
      all_output = nn.Softmax(dim=1)(all_output)
      _, predict = torch.max(all_output, 1)  
              
      # compute per-class accuracy
      matrix = confusion_matrix(all_label, torch.squeeze(predict).float(), labels= target_classes)
      per_cls_acc_vec = matrix.diagonal() / matrix.sum(axis=1) * 100
      per_cls_avg_acc = per_cls_acc_vec.mean()  # Per-class avg acc
      per_cls_acc_list = [str(np.round(i, 2)) for i in per_cls_acc_vec]
      acc_each_cls = ' '.join(per_cls_acc_list)   # str: acc of each class   
      
      # compute average accuracy
      accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
      accuracy *= 100  # overall accuracy
      
      # compute average loss
      average_loss = cumulative_loss/samples

  return  average_loss, accuracy, per_cls_avg_acc, acc_each_cls



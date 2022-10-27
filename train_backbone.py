import torch
import torch.nn.functional as F

def train_step(model, data_loader, optimizer, logger):
  samples = 0.
  cumulative_loss = 0.
  cumulative_accuracy = 0.

  # set the network to training mode: particularly important when using dropout!
  model.train() 
  
  # iterate over the training set
  for batch_idx in range(len(data_loader)):
    logger.info('Train Batch {} out of {}'.format(batch_idx+1, len(data_loader)))
    data, labels = next(iter(data_loader))
      
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

  return cumulative_loss/samples, cumulative_accuracy/samples*100



def test_step(model, data_loader, logger):
  samples = 0.
  cumulative_loss = 0.
  cumulative_accuracy = 0.

  # set the network to evaluation mode
  model.eval() 

  # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
  with torch.no_grad():

    # iterate over the test set
    for batch_idx, (data, labels) in enumerate(data_loader):
      logger.info('Test Batch {}'.format(batch_idx+1))
      # forward pass
      outputs = model(data)

      # loss computation
      loss = F.cross_entropy(outputs, labels)

      # fetch prediction and loss value
      samples+=data.shape[0]
      cumulative_loss += loss.item() # Note: the .item() is needed to extract scalars from tensors
      _, predicted = outputs.max(1)

      # compute accuracy
      cumulative_accuracy += predicted.eq(labels).sum().item()

  return cumulative_loss/samples, cumulative_accuracy/samples*100



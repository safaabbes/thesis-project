import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader

def split_dl(ds, batch_size, test_pct=0.2):
  """
  This function creates train and test DataLoaders.

  """
  random_seed = 76 #random seed insures that we always get the same split so that we always train and test on the same images
  torch.manual_seed(random_seed)
  test_size = int(test_pct * len(ds))
  train_ds, test_ds = random_split(ds, [len(ds) - test_size, test_size])

  # PyTorch Data Loaders
  train_dl = DataLoader(train_ds, batch_size, shuffle=True)
  test_dl = DataLoader(test_ds, batch_size*2) #We're increasing the batch size since we only use the test dataloader in the evaluation that requires less computation (No grad)

  return train_dl, test_dl
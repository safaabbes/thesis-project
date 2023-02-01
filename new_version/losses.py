import torch
import torch.nn as nn
import torch.nn.functional as F

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b
        

class loss_ce(nn.Module):
    '''
    Cross-entropy loss
    '''
    def __init__(self):
        super(loss_ce, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')  # reduction='none', 'sum', 'mean

    def forward(self, input, target):
        # input = F.log_softmax(input, dim=1)
        # loss = F.nll_loss(input, target, reduction='none')
        loss = self.criterion(input, target)
        return loss


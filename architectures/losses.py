import torch
import torch.nn as nn
import torch.nn.functional as F


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


class loss_op(nn.Module):
    def __init__(self, gamma=0.5):
        super(loss_op, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        # Features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss
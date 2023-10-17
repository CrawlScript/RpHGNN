# coding=utf-8

import torch.nn.functional as F

def kl_loss(y_pred, y_true):    
    y_pred = F.log_softmax(y_pred, dim=-1)
    losses = F.kl_div(y_pred, y_true, reduction='none').sum(dim=-1)
    return losses
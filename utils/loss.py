import torch
import torch.nn as nn
import torch.nn.functional as F

def operation_R(data1, data2):
    eps = torch.finfo(torch.float32).eps
    N, _, _ = data1.shape
    data1 = data1.reshape(N, 1, -1)
    data2 = data2.reshape(N, 1, -1)
    data1 = data1 - data1.mean(dim=-1, keepdim=True)
    data2 = data2 - data2.mean(dim=-1, keepdim=True)
    r = torch.sum(data1 * data2, dim=-1) / (eps + torch.sqrt(torch.sum(data1 ** 2, dim=-1)) * torch.sqrt(torch.sum(data2**2, dim=-1)))
    r = torch.clamp(r, -1., 1.)
    return r.mean()

def FusionLoss(g_hh, g_vh,d_hh, d_vh):
    return abs(operation_R(d_hh, d_vh)) / (operation_R(g_hh, g_vh) + 1.01)

def CritionLoss(preds,labels):
    criterion = nn.CrossEntropyLoss()
    return criterion(preds,labels.type(torch.long))

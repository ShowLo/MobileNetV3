# -*- coding: UTF-8 -*-

'''
Label Smoothing described in "Rethinking the Inception Architecture for Computer Vision"
Ref: https://github.com/PistonY/torch-toolbox/blob/master/torchtoolbox/nn/loss.py
'''

import torch
from torch import nn

class LabelSmoothingLoss(nn.Module):
    '''
    Label Smoothing Loss function
    '''

    def __init__(self, classes_num, label_smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing
        self.classes_num = classes_num
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        smooth_label = torch.empty(size=pred.size(), device=target.device)
        smooth_label.fill_(self.label_smoothing / self.classes_num)
        smooth_label.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-smooth_label * pred, dim=self.dim))
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class HingeLoss(_Loss):
    def __init__(self, nclass=10, weight=None, size_average=None, reduce=None,
                 reduction='elementwise_mean', pos_weight=None, squared=True):
        super(HingeLoss, self).__init__(size_average, reduce, reduction)
        self.nclass = nclass
        self.squared = squared
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        if not (target.size(0) == input.size(0)):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
        if self.pos_weight is not None:
            pos_weight = 1 + (self.pos_weight - 1) * target
        target = 2 * F.one_hot(target, num_classes=self.nclass) - 1
        target = target.float()
        loss = F.relu(1. - target * input)
        if self.squared:
            loss = 0.5 * loss ** 2
        if self.weight is not None:
            loss = loss * self.weight
        if self.pos_weight is not None:
            loss = loss * pos_weight
        loss = loss.sum(dim=-1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'elementwise_mean':
            return loss.mean()
        else:
            return loss.sum()

LOSS = {
    'ce': nn.CrossEntropyLoss,
    'hinge': HingeLoss,
}

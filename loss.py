#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()
        self.criteria = nn.MSELoss()

    def forward(self, prediction, target):
        prediction = prediction.view(-1)
        target = target.view(-1)
        loss = self.criteria(prediction, target)
        return loss



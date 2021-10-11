import os
import math
import datetime
import numpy as np
import os.path as osp
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import torch


class Trainer(object):
    """build a trainer"""
    def __init__(self, model, optimizer, criterion, device, checkpoint, max_epoch,
                 train_loader, test_loader, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.max_epoch = max_epoch
        self.checkpoint = checkpoint
        self.epoch = 0
        self.rmse_lowest = 10000
        self.pr_best = 0
        self.state_best = None

    def train(self):
        """training the model"""
        self.model.to(self.device)
        for epoch in range(self.max_epoch):
            # set training mode during the training process
            self.model.train()
            self.epoch = epoch
            for i_batch, sample_batch in enumerate(self.train_loader):
                X_data = sample_batch["data"].float().to(self.device)
                signal = sample_batch["signal"].float().to(self.device)
                pred = self.model(X_data)
                loss = self.criterion(pred, signal)
                if np.isnan(loss.item()):
                    raise ValueError('loss is nan while training')
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print("epoch/i_batch: {}/{}---loss: {:.4f}---lr: {:.5f}".format(self.epoch, i_batch,
                                                    loss.item(), self.optimizer.param_groups[0]['lr']))
            # validation and save the model with higher accuracy
            self.test()
            self.scheduler.step()

        return self.rmse_lowest, self.pr_best, self.state_best

    def test(self):
        """validate the performance of the trained model."""
        self.model.eval()
        p_all = []
        t_all = []
        for i_batch, sample_batch in enumerate(self.test_loader):
            X_data = sample_batch["data"].float().to(self.device)
            signal = sample_batch["signal"].float()
            with torch.no_grad():
                pred = self.model(X_data)
            p_all.append(pred.view(-1).data.cpu().numpy())
            t_all.append(signal.view(-1).data.numpy())
        rmse = 0
        pr = 0
        for t_one, p_one in zip(t_all, p_all):
            rmse += mean_squared_error(t_one, p_one)
            pr += pearsonr(t_one, p_one)[0]
        rmse /= len(t_all)
        pr /= len(t_all)
        if self.rmse_lowest > rmse:
            self.rmse_lowest = rmse
            self.pr_best = pr
            self.state_best = deepcopy(self.model.state_dict())
        print("rmse: {:.3f}\tpearsonr: {:.3f}\n".format(rmse, pr))


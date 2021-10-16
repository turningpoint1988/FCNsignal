#!/usr/bin/python

import os
import sys
import argparse
import random
import numpy as np
import os.path as osp
from hyperopt import hp
from hyperopt.pyll.stochastic import sample

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# custom functions defined by user
from FCNmotif import FCNA, FCNsignal, BPNet
from datasets import EPIDataSet
from trainer import Trainer
from loss import NormalLoss
torch.multiprocessing.set_sharing_strategy('file_system')


def randomparams():
    space = {
        'lr': hp.choice('lr', (0.001, 0.0001)),
        'beta1': hp.choice('beta1', (0.8, 0.9)),
        'beta2': hp.choice('beta2', (0.99, 0.999)),
        'weight': hp.choice('weight', (0, 0.001))
    }
    params = sample(space)

    return params


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="FCNsignal for predicting base-resolution binding signals")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the training data.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of a specified data.")

    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="choose gpu device. eg. '0,1,2' ")
    parser.add_argument("-s", dest="seed", type=int, default=5,
                        help="Random seed to have reproducible results.")
    # Arguments for Adam optimization
    parser.add_argument("-b", dest="batch_size", type=int, default=500,
                        help="Number of sequences sent to the network in one step.")
    parser.add_argument("-e", dest="max_epoch", type=int, default=50,
                        help="Number of training steps.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
        torch.manual_seed(args.seed)
    motifLen = 16
    Data = np.load(osp.join(args.data_dir, '%s_train.npz' % args.name))
    seqs, signals = Data['data'], Data['signal']
    ratio = 0.1
    number_t = int(len(seqs) * ratio)
    index = list(range(len(seqs)))
    index_val = random.sample(index, number_t)
    index_train = list(set(index) - set(index_val))
    # build training data generator
    data_tr = seqs[index_train]
    signal_tr = signals[index_train]
    train_data = EPIDataSet(data_tr, signal_tr)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # build val data generator
    data_va = seqs[index_val]
    signal_va = signals[index_val]
    val_data = EPIDataSet(data_va, signal_va)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
    rmse_lowest = 10000
    pr_best = 0
    for trial in range(15):
        params = randomparams()
        model = FCNsignal(motiflen=motifLen)
        # model = BPNet()
        # model = FCNA(motiflen=motifLen)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']), weight_decay=params['weight'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        criterion = NormalLoss()
        executor = Trainer(model=model,
                           optimizer=optimizer,
                           criterion=criterion,
                           device=device,
                           checkpoint=args.checkpoint,
                           max_epoch=args.max_epoch,
                           train_loader=train_loader,
                           test_loader=val_loader,
                           scheduler=scheduler)

        rmse, pr, state_dict = executor.train()
        if rmse_lowest > rmse:
            rmse_lowest = rmse
            pr_best = pr
            checkpoint_file = osp.join(args.checkpoint, 'model_best.pth')
            torch.save({
                'model_state_dict': state_dict
            }, checkpoint_file)


if __name__ == "__main__":
    main()


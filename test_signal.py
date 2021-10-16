#!/usr/bin/python

import os
import sys
import argparse
import numpy as np
import os.path as osp

import torch
from torch.utils.data import DataLoader

# custom functions defined by user
from FCNmotif import BPNet, FCNA, FCNsignal
from datasets import EPIDataSet
from trainer import Trainer
from loss import NormalLoss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr
torch.multiprocessing.set_sharing_strategy('file_system')


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="FCNsignal for testing data")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the test data.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of a specified data.")

    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    if torch.cuda.is_available():
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
    f = open(osp.join(args.checkpoint, 'record.txt'), 'w')
    f.write('RMSE\tPR\n')
    motifLen = 16
    Data = np.load(osp.join(args.data_dir, '%s_test.npz' % args.name))
    seqs, signals = Data['data'], Data['signal']
    test_data = EPIDataSet(seqs, signals)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    #
    Data = np.load(osp.join(args.data_dir, '%s_neg.npz' % args.name))
    seqs, signals = Data['data'], Data['signal']
    neg_data = EPIDataSet(seqs, signals)
    neg_loader = DataLoader(neg_data, batch_size=1, shuffle=False, num_workers=0)
    # Load weights
    checkpoint_file = osp.join(args.checkpoint, 'model_best.pth')
    chk = torch.load(checkpoint_file)
    state_dict = chk['model_state_dict']
    model = FCNsignal(motiflen=motifLen)
    # model = BPNet()
    # model = FCNA(motiflen=motifLen)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    p_all = []
    t_all = []
    max_pos = []
    max_pos_t = []
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(device)
        signal = sample_batch["signal"].float()
        with torch.no_grad():
            pred = model(X_data)
        signal_p = pred.view(-1).data.cpu().numpy()
        p_all.append(signal_p)
        max_pos.append(np.max(signal_p))
        signal_t = signal.view(-1).data.cpu().numpy()
        t_all.append(signal_t)
        max_pos_t.append(np.max(signal_t))
    label_pos = np.ones(len(max_pos_t))

    rmse = 0
    pr = 0
    records1 = []
    records2 = []
    for t_one, p_one in zip(t_all, p_all):
        rmse += mean_squared_error(t_one, p_one)
        records1.append(mean_squared_error(t_one, p_one))
        pr += pearsonr(t_one, p_one)[0]
        records2.append(pearsonr(t_one, p_one)[0])
    rmse /= len(t_all)
    pr /= len(t_all)
    print("{}: {:.3f}\t{:.3f}\n".format(args.name, rmse, pr))
    f.write("{:.3f}\t{:.3f}\n".format(rmse, pr))
    ##
    p_all = []
    t_all = []
    max_neg = []
    max_neg_t = []
    for i_batch, sample_batch in enumerate(neg_loader):
        X_data = sample_batch["data"].float().to(device)
        signal = sample_batch["signal"].float()
        with torch.no_grad():
            pred = model(X_data)
        signal_p = pred.view(-1).data.cpu().numpy()
        p_all.append(signal_p)
        max_neg.append(np.max(signal_p))
        signal_t = signal.view(-1).data.cpu().numpy()
        t_all.append(signal_t)
        max_neg_t.append(np.max(signal_t))
    label_neg = np.zeros(len(max_neg_t))
    rmse = 0
    pr = 0
    records1 = []
    records2 = []
    for t_one, p_one in zip(t_all, p_all):
        rmse += mean_squared_error(t_one, p_one)
        records1.append(mean_squared_error(t_one, p_one))
        pr += pearsonr(t_one, p_one)[0]
        records2.append(pearsonr(t_one, p_one)[0])
    rmse /= len(t_all)
    pr /= len(t_all)
    print("{}: {:.3f}\t{:.3f}\n".format(args.name, rmse, pr))
    # f.write("{:.3f}\t{:.3f}\n".format(rmse, pr))
    label = np.concatenate((label_pos, label_neg))
    pred = np.concatenate((np.array(max_pos), np.array(max_neg)))
    #
    auroc = roc_auc_score(label, pred)
    auprc = average_precision_score(label, pred)
    print("{}: {:.3f}\t{:.3f}\n".format(args.name, auroc, auprc))
    f.write('AUC\tPRAUC\n')
    f.write("{:.3f}\t{:.3f}\n".format(auroc, auprc))
    f.close()


if __name__ == "__main__":
    main()


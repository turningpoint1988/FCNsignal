#!/usr/bin/python

import os
import sys
import time
import argparse
import math
import numpy as np
import os.path as osp

import torch
from torch.utils.data import DataLoader
# custom functions defined by user
from FCNmotif import FCNAGRU
from datasets import EPIDataSet
from utils import Dict
torch.multiprocessing.set_sharing_strategy('file_system')


def extract(signal, thres):
    start = end = -1
    position = np.argmax(signal)
    Max = np.max(signal)
    if Max > thres:
        start = position - WINDOW // 2
        end = position + WINDOW // 2
        if start < 0:
            start = 0
            end = start + WINDOW
        if end >= len(signal):
            end = len(signal) - 1
            start = end - WINDOW
    if Max < thres:
        start = 0
        end = 0

    return int(start), int(end)


def locating(device, model, state_dict, test_loader, outdir, thres, name):
    # loading model parameters
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    f1 = open(osp.join(outdir, '{}_neg.bed'.format(name)), 'w')
    f2 = open(osp.join(outdir, '{}_pos.bed'.format(name)), 'w')
    num_pos = 0
    num_neg = 0
    # for test data
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(device)
        X_bed = sample_batch["signal"].int().data.numpy()[0]
        with torch.no_grad():
            signal_p = model(X_data)
        signal_p = signal_p.view(-1).data.cpu().numpy()
        start, end = extract(signal_p, thres)
        if start == 0 and end == 0:
            # print("The sequence is predicted to be a negative sample.")
            num_neg += 1
            f1.write("{}\t{}\t{}\n".format('chr17', X_bed[0], X_bed[1]))
        elif start > 0 and end > 0:
            start_o = X_bed[0]
            start += start_o
            end += start_o
            # print("The sequence is predicted to be a positive sample.")
            num_pos += 1
            f2.write("{}\t{}\t{}\n".format('chr17', start, end))
    num_all = num_pos + num_neg
    print("The number of positive and negative samples is {} ({:.3f}) and {} ({:.3f})".format(num_pos, num_pos/num_all, num_neg, num_neg/num_all))
    f1.close()
    f2.close()


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="FCN for motif location")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the training data.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of a specified data.")

    parser.add_argument("-t", dest="thres", type=float, default=0.5,
                        help="threshold value.")
    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")
    parser.add_argument("-o", dest="outdir", type=str, default='./fasta/',
                        help="Where to save experimental results.")

    return parser.parse_args()


args = get_args()
motifLen = 16
WINDOW = 60


def main():
    """Create the model and start the training."""
    if torch.cuda.is_available():
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
    name = args.name
    Data = np.load(osp.join(args.data_dir, 'chromosome/data/chr17.npz'))
    seqs_data, seqs_bed = Data['data'], Data['bed']
    
    data_te = EPIDataSet(seqs_data, seqs_bed)
    test_loader = DataLoader(data_te, batch_size=1, shuffle=False, num_workers=0)
    # Load weights
    checkpoint_file = osp.join(args.checkpoint, 'model_best.pth')
    chk = torch.load(checkpoint_file)
    state_dict = chk['model_state_dict']
    model = FCNAGRU(motiflen=motifLen)
    locating(device, model, state_dict, test_loader, args.outdir, args.thres, name)


if __name__ == "__main__":
    main()


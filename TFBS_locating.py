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
from FCNmotif import FCNA, FCNsignal
from datasets import EPIDataSet
torch.multiprocessing.set_sharing_strategy('file_system')


def Rencode(seq_onehot):
    seq = ''
    num = seq_onehot.shape[1]
    for i in range(num):
        character = seq_onehot[:, i]
        if np.sum(character) == 0:
            seq += 'N'
            continue
        index = np.argmax(character)
        if index == 0:
            seq += 'A'
        elif index == 1:
            seq += 'C'
        elif index == 2:
            seq += 'G'
        elif index == 3:
            seq += 'T'

    return seq


def extract(signal, thres):
    start = end = 0
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

    return int(start), int(end)


def locating(device, model, state_dict, test_loader, outdir, thres=0.3):
    # loading model parameters
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    f1 = open(osp.join(outdir, 'test_original.fasta'), 'w')
    f2 = open(osp.join(outdir, 'test_located.fasta'), 'w')
    # for test data
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(device)
        with torch.no_grad():
            signal_p = model(X_data)
        signal_p = signal_p.view(-1).data.cpu().numpy()
        start, end = extract(signal_p, thres)
        if start == end: continue
        data = X_data[0].data.cpu().numpy()
        seq_orig = Rencode(data)
        data_loc = data[:, start:end]
        seq_loc = Rencode(data_loc)
        if 'N' in seq_loc:
            continue
        f1.write('>seq{}\n'.format(i_batch))
        f1.write('{}\n'.format(seq_orig))
        f2.write('>seq{}\n'.format(i_batch))
        f2.write('{}\n'.format(seq_loc))

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
WINDOW = 200


def main():
    """Create the model and start the training."""
    if torch.cuda.is_available():
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
    
    Data = np.load(osp.join(args.data_dir, '%s_test.npz' % args.name))
    seqs_te, signals_te = Data['data'], Data['signal']
    
    data_te = EPIDataSet(seqs_te, signals_te)
    test_loader = DataLoader(data_te, batch_size=1, shuffle=False, num_workers=0)
    # Load weights
    checkpoint_file = osp.join(args.checkpoint, 'model_best.pth')
    chk = torch.load(checkpoint_file)
    state_dict = chk['model_state_dict']
    model = FCNsignal(motiflen=motifLen)
    locating(device, model, state_dict, test_loader, args.outdir, args.thres)


if __name__ == "__main__":
    main()


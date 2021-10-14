#!/usr/bin/python

import os
import sys
import time
import argparse
import math
import numpy as np
import os.path as osp
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

# custom functions defined by user
from datasets import EPIDataSet
from utils import Dict
import torch.nn as nn
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')


def upsample(x, out_size):
    return F.interpolate(x, size=out_size, mode='linear', align_corners=False)


def bn_relu_conv(in_, out_, kernel_size=3, stride=1, bias=False):
    padding = kernel_size // 2
    return nn.Sequential(nn.BatchNorm1d(in_),
                         nn.ReLU(inplace=True),
                         nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))


class FCNAO(nn.Module):
    """FCN for motif mining"""
    def __init__(self, motiflen=16):
        super(FCNAO, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.aap = nn.AdaptiveAvgPool1d(1)
        # decode process
        self.blend4 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, 4, kernel_size=3)
        self.blend1 = bn_relu_conv(4, 1, kernel_size=3)
        # general functions
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        score = out1
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        skip4 = out1
        up5 = self.aap(out1)
        # decode process
        up4 = upsample(up5, skip4.size()[-1])
        up4 = up4 + skip4
        up4 = self.blend4(up4)
        up3 = upsample(up4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        up1 = up1 + skip1
        out_dense = self.blend1(up1)
        # out_dense = self.sigmoid(out_dense)

        return out_dense[0], score[0]


class FCNAGRU(nn.Module):
    """FCN for motif mining"""
    def __init__(self, motiflen=16):
        super(FCNAGRU, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        # self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        # decode process
        # self.gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.gru1 = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.gru2 = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.gru_drop = nn.Dropout(p=0.2)
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.blend4 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend1 = bn_relu_conv(64, 1, kernel_size=3)
        # general functions
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ELU(alpha=0.1, inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        score = out1
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        out1 = out1.permute(0, 2, 1)
        # out1_1, _ = self.gru(out1)
        # out1_2, _ = self.gru(torch.flip(out1, [1]))
        out1_1, _ = self.gru1(out1)
        out1_2, _ = self.gru2(torch.flip(out1, [1]))
        out1 = out1_1 + out1_2
        out1 = self.gru_drop(out1)
        skip4 = out1.permute(0, 2, 1)
        up5 = self.aap(skip4)
        # decode process
        up4 = upsample(up5, skip4.size()[-1])
        up4 = up4 + skip4
        up4 = self.blend4(up4)
        up3 = upsample(up4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        out_dense = self.blend1(up1)
        # out_dense = self.sigmoid(out_dense)

        return out_dense[0], score[0]


class BPNet(nn.Module):
    """building BPNet on the Pytorch platform."""
    def __init__(self, motiflen=25, batchnormalization=False):
        super(BPNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=motiflen, padding=motiflen // 2)
        self.relu1 = nn.ReLU(inplace=True)
        # sequential model
        self.sequential_model = nn.ModuleList()
        for i in range(1, 10):
            if batchnormalization:
                self.sequential_model.append((nn.Sequential(
                    nn.BatchNorm1d(64),
                    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2**i, dilation=2**i),
                    nn.ReLU(inplace=True))))
            else:
                self.sequential_model.append((nn.Sequential(
                    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2**i, dilation=2**i),
                    nn.ReLU(inplace=True))))
        self.convtranspose1 = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=motiflen, padding=motiflen // 2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):

        """Construct a new computation graph at each froward"""
        b, c, l = data.size()
        conv1_copy = deepcopy(self.conv1)
        conv1_copy.padding = 0
        score = conv1_copy(data)
        x = self.conv1(data)
        x = self.relu1(x)
        for module in self.sequential_model:
            conv_x = module(x)
            x = conv_x + x
        bottleneck = x
        out = self.convtranspose1(bottleneck)

        return out[0], score[0]
        

def extract(signal, thres):
    start = end = 0
    seqLen = len(signal)
    position = np.argmax(signal)
    Max = np.max(signal)
    if Max > thres:
        start = position - WINDOW // 2
        end = position + WINDOW // 2
        if start < 0: start = 0
        if end > seqLen - 1:
            end = seqLen - 1
            start = end - WINDOW

    return int(start), int(end)


def motif_all(device, model, state_dict, test_loader, outdir, thres=0.5):
    # loading model parameters
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    # for test data
    motif_data = [0.] * kernel_num
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(device)
        with torch.no_grad():
            signal_p, score_p = model(X_data)
        signal_p = signal_p.view(-1).data.cpu().numpy()
        start, end = extract(signal_p, thres)

        if start == end: continue
        data = X_data[0].data.cpu().numpy()
        score_p = score_p.data.cpu().numpy()
        score_p = score_p[:, start:end]
        max_index = np.argmax(score_p, axis=1)
        for i in range(kernel_num):
            index = max_index[i]
            index += start
            data_slice = data[:, index:(index + motifLen)]
            motif_data[i] += data_slice

    pfms = compute_pfm(motif_data, k=kernel_num)
    writeFile(pfms, 'motif', outdir)


def motif_one(device, model, state_dict, test_loader, outdir, thres):
    # loading model parameters
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    # for test data
    motif_data = [0.] * kernel_num
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(device)
        signal = sample_batch["signal"].float()
        with torch.no_grad():
            signal_p, score_p = model(X_data)
        signal_p = signal_p.view(-1).data.cpu().numpy()
        start, end = extract(signal_p, thres)
        if start == end: continue
        data = X_data[0].data.cpu().numpy()
        score_p = score_p.data.cpu().numpy()
        score_p = score_p[:, start:end]
        max_row = np.max(score_p, axis=1)
        max_row_index = np.argmax(score_p, axis=1)
        kernel_index = np.argmax(max_row)
        index = max_row_index[kernel_index]
        index += start
        data_slice = data[:, index:(index + motifLen)]
        motif_data[kernel_index] += data_slice

    pfms = compute_pfm(motif_data, k=kernel_num)
    writeFile(pfms, 'motif', outdir)


def compute_pfm(motifs, k):
    pfms = []
    informations = []
    for motif in motifs:
        if np.sum(motif) == 0.: continue
        sum_ = np.sum(motif, axis=0)
        if sum_[0] < 10: continue
        pfm = motif / sum_
        pfms.append(pfm)
        #
        row, col = pfm.shape
        information = 0
        for j in range(col):
            information += 2 + np.sum(pfm[:, j] * np.log2(pfm[:, j]+1e-8))
        informations.append(information)
    pfms_filter = []
    index = np.argsort(np.array(informations))
    index = index[::-1]
    for i in range(len(informations)):
        index_c = index[i]
        pfms_filter.append(pfms[index_c])
    return pfms_filter


def writeFile(pfm, flag, outdir):
    out_f = open(outdir + '/{}.meme'.format(flag), 'w')
    out_f.write("MEME version 5.3.3\n\n")
    out_f.write("ALPHABET= ACGT\n\n")
    out_f.write("strands: + -\n\n")
    out_f.write("Background letter frequencies\n")
    out_f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
    for i in range(len(pfm)):
        out_f.write("MOTIF " + "{}\n".format(i+1))
        out_f.write("letter-probability matrix: alength= 4 w= {} nsites= {}\n".format(motifLen, motifLen))
        current_pfm = pfm[i]
        for col in range(current_pfm.shape[1]):
            for row in range(current_pfm.shape[0]):
                out_f.write("{:.4f} ".format(current_pfm[row, col]))
            out_f.write("\n")
        out_f.write("\n")
    out_f.close()


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
    parser.add_argument("-o", dest="outdir", type=str, default='./motifs/',
                        help="Where to save experimental results.")

    return parser.parse_args()


args = get_args()
motifLen = 16 #25
WINDOW = 100
kernel_num = 64


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
    seqs, signals = Data['data'], Data['signal']

    # build test data generator
    data_te = seqs
    signals_te = signals
    test_data = EPIDataSet(data_te, signals_te)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    # Load weights
    checkpoint_file = osp.join(args.checkpoint, 'model_best.pth')
    chk = torch.load(checkpoint_file)
    state_dict = chk['model_state_dict']
    # model = FCNAO(motiflen=motifLen)
    # model = BPNet()
    model = FCNAGRU(motiflen=motifLen)
    motif_all(device, model, state_dict, test_loader, args.outdir, args.thres)


if __name__ == "__main__":
    main()


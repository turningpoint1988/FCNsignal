# coding:utf-8
import os.path as osp
import os
import sys
import argparse
import itertools
import random
import numpy as np
from Bio import SeqIO
import pyBigWig

SEQ_LEN = 1001
OFFSET = 3000
INDEX = ['chr' + str(i + 1) for i in range(23)]
INDEX[22] = 'chrX'
SMOOTH = True
WINDOW = 15
THRESHOLD = 0.05
CHROM = {}
with open('/your_path/genome/chromsize') as f:
    for i in f:
        line_split = i.strip().split()
        CHROM[line_split[0]] = int(line_split[1])


def one_hot(sequence_dict, chrom, start, end):
    seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
                'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
                'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
    temp = []
    seq = str(sequence_dict[chrom].seq[start:end])
    for c in seq:
        temp.append(seq_dict.get(c, [0, 0, 0, 0]))
    return temp


def GetbigWig(file, chrom, start, end):
    bw = pyBigWig.open(file)
    sample = np.array(bw.values(chrom, start, end))
    bw.close()
    return sample


def Smoothavg(x, window):
    avg =[]
    for j in range(len(x)-window+1):
        avg.append(np.mean(x[j:(j+window)]))

    return np.array(avg)


# random position for the max signal
def pos_location(chrom, start, end):
    original_len = end - start
    if original_len < SEQ_LEN:
        start_update = start - np.ceil((SEQ_LEN - original_len) / 2)
    elif original_len > SEQ_LEN:
        start_update = start + np.ceil((original_len - SEQ_LEN) / 2)
    else:
        start_update = start

    start_update += np.random.randint(-100, 100)
    end_update = start_update + SEQ_LEN
    if end_update > CHROM[chrom]:
        end_update = CHROM[chrom]
        start_update = end_update - SEQ_LEN
    return int(start_update), int(end_update)


# upstream of positive peaks
def neg_location(chrom, start, end):
    start_update = start - OFFSET
    if start_update < 0:
        start_update = 0
    end_update = start_update + SEQ_LEN

    return int(start_update), int(end_update)


def get_data_local_neg(seqs_bed, sequence_dict, signal_file, max_neg):
    seqs_neg = []
    signals_neg = []
    lines = open(seqs_bed).readlines()
    index = list(range(len(lines)))
    for i in index:
        line_split = lines[i].strip().split()
        chrom = line_split[0]
        if chrom not in INDEX:
            continue
        start, end = int(line_split[1]), int(line_split[2])
        # for negative peaks
        start_n, end_n = neg_location(chrom, start, end)
        # for positive peaks
        signal = GetbigWig(signal_file, chrom, start_n, end_n)
        signal[np.isnan(signal)] = 0.
        if max_neg <= max(signal):
            continue
        seqs_neg.append(one_hot(sequence_dict, chrom, start_n, end_n))
        signals_neg.append(signal)
    #
    seqs_neg = np.array(seqs_neg, dtype=np.float32)
    seqs_neg = seqs_neg.transpose((0, 2, 1))
    signals_neg = [[x] for x in signals_neg]
    signals_neg = np.array(signals_neg, dtype=np.float32)

    return seqs_neg, signals_neg


def get_data_local(seqs_bed, sequence_dict, signal_file):
    seqs = []
    signals = []
    max_all = []
    lines = open(seqs_bed).readlines()
    index = list(range(len(lines)))
    if SMOOTH:
        print("Using the smooth way.")
    else:
        print("Using the normal way.")
    for i in index:
        line_split = lines[i].strip().split()
        chrom = line_split[0]
        if chrom not in INDEX:
            continue
        start, end = int(line_split[1]), int(line_split[2])
        start_p, end_p = pos_location(chrom, start, end)
        if SMOOTH:
            signal = GetbigWig(signal_file, chrom, start_p - WINDOW // 2, end_p + WINDOW // 2)
            signal[np.isnan(signal)] = 0.
            signal = Smoothavg(signal, WINDOW)
            Max = np.max(signal)
        else:
            signal = GetbigWig(signal_file, chrom, start_p, end_p)
            signal[np.isnan(signal)] = 0.
            Max = np.max(signal)
        if Max == 0: continue
        max_all.append(Max)
        signals.append(signal)
        seqs.append(one_hot(sequence_dict, chrom, start_p, end_p))

    print("The minimum and maximum are {:.3f} and {:.3f}".format(np.min(max_all), np.max(max_all)))
    sort_i = np.argsort(max_all)
    start_i = int(THRESHOLD*len(sort_i))
    seqs_filter = []
    signals_filter = []
    for i in sort_i[start_i:]:
        seqs_filter.append(seqs[i])
        signals_filter.append(signals[i])
    seqs_filter = np.array(seqs_filter, dtype=np.float32)
    seqs_filter = seqs_filter.transpose((0, 2, 1))
    signals_filter = [[x] for x in signals_filter]
    signals_filter = np.array(signals_filter, dtype=np.float32)

    return seqs_filter, signals_filter, max_all[sort_i[start_i]]


def get_args():
    parser = argparse.ArgumentParser(description="pre-process data.")
    parser.add_argument("-d", dest="dir", type=str, default='')
    parser.add_argument("-n", dest="name", type=str, default='')
    parser.add_argument("-s", dest="seed", type=int, default=666, help="Random seed to have reproducible results.")

    return parser.parse_args()


def main():
    params = get_args()
    random.seed(params.seed)
    name = params.name
    data_dir = params.dir
    out_dir = osp.join(params.dir, 'data')
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open('/your_path/genome/hg38.fa'), 'fasta'))
    print('Experiment on %s dataset' % name)
    seqs_bed = data_dir + '/thresholded.bed'
    pvalue_file = data_dir + '/p-value.bigWig'
    seqs, signals, max_neg = get_data_local(seqs_bed, sequence_dict, pvalue_file)
    signals = np.log10(1+signals)
    ratio = 0.2
    number_t = int(len(seqs) * ratio)
    index = list(range(len(seqs)))
    index_test = random.sample(index, number_t)
    seqs_te = seqs[index_test]
    signals_te = signals[index_test]
    index_train = list(set(index) - set(index_test))
    seqs_tr = seqs[index_train]
    signals_tr = signals[index_train]
    labels_tr = np.ones((len(seqs_tr), 1))

    labels_te = np.ones((len(seqs_te), 1))
    np.savez(out_dir + '/%s_test.npz' % name, data=seqs_te, signal=signals_te, label=labels_te)    
    ## selecting negative samples ##
    seqs_neg, signals_neg = get_data_local_neg(seqs_bed, sequence_dict, pvalue_file, max_neg)
    signals_neg = np.log10(1+signals_neg)
    ratio = 0.2
    number_t = int(len(seqs_neg) * ratio)
    index_neg = list(range(len(seqs_neg)))
    index_neg_test = random.sample(index_neg, number_t)
    seqs_neg_te = seqs_neg[index_neg_test]
    signals_neg_te = signals_neg[index_neg_test]
    index_neg_train = list(set(index_neg) - set(index_neg_test))
    seqs_neg_tr = seqs_neg[index_neg_train]
    signals_neg_tr = signals_neg[index_neg_train]

    labels_neg_tr = np.zeros((len(seqs_neg_tr), 1))
    labels_neg_te = np.zeros((len(seqs_neg_te), 1))
    np.savez(out_dir + '/%s_neg.npz' % name, data=seqs_neg_te, signal=signals_neg_te, label=labels_neg_te)

    seqs_total = np.concatenate((seqs_tr, seqs_neg_tr), axis=0)
    signals_total = np.concatenate((signals_tr, signals_neg_tr))
    labels_total = np.concatenate((labels_tr, labels_neg_tr), axis=0)
    np.savez(out_dir + '/%s_train.npz' % name, data=seqs_total, signal=signals_total, label=labels_total)
    print("{}: The train data are: {}".format(name, len(seqs_total)))
    print("{}: The test data are: {}".format(name, len(seqs_te)))
    print("{}: The negative data are: {}\n".format(name, len(seqs_neg_te)))
    #


if __name__ == '__main__':  main()

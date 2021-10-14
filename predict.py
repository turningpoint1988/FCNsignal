#!/usr/bin/python

import os
import sys
import numpy as np
import os.path as osp
from Bio import SeqIO

import torch
# custom functions defined by user
from FCNmotif import FCNAGRU
torch.multiprocessing.set_sharing_strategy('file_system')
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns

curDir = os.getcwd()
sys.path.append(curDir+'/'+'webserver')
#sys.path.append('/home/zqh/web/iDRBP_MMC/iDRBP_MMC/webserver')
# from webserver import send_email


def pieplot(number1, number2, name, outdir):

    fig = plt.figure(figsize=(8, 10))
    sns.set_theme(style="whitegrid")
    labels = ['Supported', 'Unsupported']
    sizes = [number1 / (number1 + number2), number2 / (number1 + number2)]
    explode = (0, 0)
    plt.pie(sizes, explode=explode, labels=labels, labeldistance=0.5, pctdistance=0.3, autopct='%1.1f%%',
            shadow=False, startangle=90, textprops={'fontsize': 12, 'color': 'black'}, radius=1)
    plt.axis('equal')
    plt.legend(loc='lower center', fontsize=12)
    plt.title("{}({} potential binding regions are found)".format(name, number1 + number2), fontsize=18)

    plt.savefig(osp.join(outdir, "{}.png".format(name)), format='png')
    plt.close(fig)


def lineplot(signals, name, outdir, thres):
    for key in signals.keys():
        signal = signals[key]
        segment_len = 1000
        length = len(signal)
        n = int(np.ceil(length / segment_len))
        positions_x = []
        positions_y = []
        for i in range(n):
            segment = signal[i * segment_len:(i + 1) * segment_len]
            position = np.argmax(segment) + i * segment_len
            Max = np.max(segment)
            if Max > thres:
                positions_x.append(position)
                positions_y.append(np.min(segment) + 0.2)
        x = np.asarray(list(range(length))) + 1
        df = pd.DataFrame({'x': x, 'Prediction': signal})
        fig = plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        sns.lineplot(x="x", y="Prediction", data=df)
        for position_x, position_y in zip(positions_x, positions_y):
            plt.axvline(x=position_x, ls="--", c="green")
            plt.text(position_x, position_y, str(position_x))
        plt.xlabel(key, fontsize=15)
        plt.ylabel("Prediction", fontsize=15)
        plt.tick_params(labelsize=12)
        plt.title("{}".format(name), fontsize=18)

        plt.savefig(osp.join(outdir, "{}.png".format(key)), format='png')
        plt.close(fig)


def one_hot(sequence_dict, chrom, start, end):
    seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
                'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
                'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
    temp = []
    seq = str(sequence_dict[chrom].seq[start:end])
    for c in seq:
        temp.append(seq_dict.get(c, [0, 0, 0, 0]))
    temp = np.array([temp], dtype=np.float32)

    return temp.transpose((0, 2, 1))


def extract(signal, thres, Window):
    segment_len = 1000
    length = len(signal)
    n = int(np.ceil(length / segment_len))
    segments = []
    for i in range(n):
        segments.append(signal[i*segment_len:(i+1)*segment_len])
    starts = []
    ends = []
    for i in range(len(segments)):
        segment = segments[i]
        position = np.argmax(segment) + i*segment_len
        Max = np.max(segment)
        if Max > thres:
            start = int(position - Window // 2)
            end = int(position + Window // 2)
        else:
            start = 0
            end = 0
        starts.append(start)
        ends.append(end)
    return starts, ends


def locating(parameters, sequence_dict, model, beds, outdir):
    # loading model parameters
    model.eval()
    beds_pos = []
    signals = {}
    num_pos = 0
    f = open(osp.join(outdir, 'predictions.txt'), 'w')
    for i, bed in enumerate(beds):
        chrom = bed.strip().split()[0]
        start = int(bed.split()[1])
        end = int(bed.split()[2])
        X_data = torch.from_numpy(one_hot(sequence_dict, chrom, start, end))
        X_data = X_data.float().to(parameters['Device'])
        with torch.no_grad():
            signal_p = model(X_data)
        signal_p = signal_p.view(-1).data.cpu().numpy()
        f.write('\t'.join([str(round(x, 3)) for x in signal_p]) + '\n')
        signals["{}:{}-{}".format(chrom, start, end)] = signal_p
        starts, ends = extract(signal_p, parameters['Threshold'], parameters['Window'])
        for start_p, end_p in zip(starts, ends):
            if start_p == 0 and end_p == 0:
                continue
            else:
                start_p += start
                end_p += start
                # print("The sequence is predicted to be a positive sample.")
                num_pos += 1
                beds_pos.append((chrom, start_p, end_p))
    print("A total of {} potential binding regions are found.".format(num_pos))
    f.close()
    return beds_pos, signals


def intersection(pred_beds, ref_beds, outdir):
    ref_dict = {}
    with open(ref_beds) as f:
        for line in f:
            line_split = line.strip().split()
            chrom = line_split[0]
            start = int(line_split[1])
            end = int(line_split[2])
            if chrom not in ref_dict.keys():
                ref_dict[chrom] = [(start, end)]
            else:
                ref_dict[chrom].append((start, end))
    number1 = 0
    number2 = 0
    f1 = open(osp.join(outdir, 'supported.bed'), 'w')
    f2 = open(osp.join(outdir, 'unsupported.bed'), 'w')
    for bed in pred_beds:
        chrom = bed[0]
        start = bed[1]
        end = bed[2]
        reference = ref_dict[chrom]
        flag = False
        for ref in reference:
            if start <= ref[0] < end or ref[0] <= start < ref[1]:
                flag = True
                break
        if flag:
            number1 += 1
            f1.write("{}\t{}\t{}\n".format(chrom, start, end))
        else:
            number2 += 1
            f2.write("{}\t{}\t{}\n".format(chrom, start, end))
    f1.close()
    f2.close()
    return number1, number2


def main():
    root = "/home/zqh/FCNAsignal_TF/website"
    infile = osp.join(root, "input.bed")
    window = 100
    name = 'CTCF'
    threshold = 1.5
    model_file = osp.join(root, "models/{}_model.pth".format(name))
    outdir = osp.join(root, "outputs")
    genome = 'hg38'
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(osp.join(root, "genomes/{}.fa".format(genome))), 'fasta'))
    references = osp.join(root, "references/{}/{}_sorted.bed".format(name, name))
    device = torch.device("cuda:0")
    with open(infile) as f:
        beds = f.readlines()
    # Load weights
    chk = torch.load(model_file)
    state_dict = chk['model_state_dict']
    model = FCNAGRU(motiflen=16)
    model.load_state_dict(state_dict)
    model.to(device)
    parameters = {'Window': window, 'Name': name, 'Threshold': threshold, 'Device': device}
    beds_pos, signal_p = locating(parameters, sequence_dict, model, beds, outdir)
    number1, number2 = intersection(beds_pos, references, outdir)
    pieplot(number1, number2, name, outdir)
    lineplot(signal_p, name, outdir, parameters['Threshold'])


def web_service(root, user_email, window, name, threshold, genome):
    # root = "/home/zqh/FCNAsignal_TF/website"
    # window = 60
    # name = 'CTCF'
    # threshold = 1.5
    # genome = 'hg38'
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(osp.join(root, "genomes/{}.fa".format(genome))), 'fasta'))
    references = osp.join(root, "references/{}/{}_sorted.bed".format(name, name))
    device = torch.device("cpu")
    infile = osp.join(root, "input.bed")
    # test_fasta = os.path.join(user_dir, 'test.fasta')  # 输入文件
    # test_fasta = test_fasta.replace('\\','/')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!', infile)
    outdir = osp.join(root, "outputs")
    os.mkdir(outdir)
    # Load weights
    model_file = osp.join(root, "models/{}_model.pth".format(name))
    # model_file = 'C:\\Users\\Administrator\\Desktop\\web1\\iDRBP_MMC\\iDRBP_MMC\\webserver\\model\\' + family_name + '\\model_best0.pth'
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = checkpoint['model_state_dict']
    model = FCNAGRU(motiflen=16)
    model.load_state_dict(state_dict)
    model.to(device)
    parameters = {'Window': window, 'Name': name, 'Threshold': threshold, 'Device': device}
    beds_pos, signal_p = locating(parameters, sequence_dict, model, beds, outdir)
    number1, number2 = intersection(beds_pos, references, outdir)
    pieplot(number1, number2, name, outdir)
    lineplot(signal_p, name, outdir, parameters['Threshold'])
    send_email.send_email_1(user_dir.split('/')[-1], user_email)


if __name__ == "__main__":
    main()


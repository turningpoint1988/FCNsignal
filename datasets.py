import os
import os.path as osp
import numpy as np
import torch
from torch.utils import data

__all__ = ['EPIDataSet']


class EPIDataSet(data.Dataset):
    def __init__(self, data, signal):
        super(EPIDataSet, self).__init__()
        self.data = data
        self.signal = signal

        assert len(self.data) == len(self.signal), \
            "the number of sequences and labels must be consistent."

        print("The number of data is {}".format(len(self.signal)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, index):
        data_one = self.data[index]
        signal_one = self.signal[index]

        return {"data": data_one, "signal": signal_one, "idx": index}




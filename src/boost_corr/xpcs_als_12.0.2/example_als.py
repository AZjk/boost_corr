import struct
import numpy as np
import time
import logging
import os
from tqdm import trange, tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from xpcs_boost import XpcsBoost as XB
from imm_handler import ImmDataset
from xpcs_metadata import XpcsMetaData
from xpcs_dataset import XpcsDataset
import matplotlib.pyplot as plt

import glob2


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-24s: %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)


class ALSDataset2(XpcsDataset):
    """
    Parameters
    ----------
    filename: string
        path to .imm file
    frames_per_point: integer
        number of frames to return as one datum
    """
    def __init__(
        self,
        scaling=1E4,
        dtype=np.uint8,
        *args,
        **kwargs,
    ):
        super(ALSDataset, self).__init__(*args, dtype=dtype, **kwargs)
        self.dataset_type = "ALS Timepix"
        self.scaling = scaling
        self.cache = self.read_data()
 
    def read_data(self):
        """
        read table of content for IMM datasets
        """
        x = np.loadtxt(self.fname, dtype=np.int64, skiprows=1)
        x = (x - np.min(x)) // int(self.scaling)
        print(np.min(x), np.max(x), x.dtype)
        data = np.bincount(x).reshape(-1, 1)
        data = data.astype(np.float32)
        print(data.shape, np.min(data), np.max(data), data[0])
        self.update_batch_info(data.size)
        return torch.tensor(data)
 
    def __getitem__(self, index):
        beg, end = self.get_raw_index(index)
        return self.cache[beg: end]


class ALSDataset(XpcsDataset):
    """
    Parameters
    ----------
    filename: string
        path to .imm file
    frames_per_point: integer
        number of frames to return as one datum
    """
    def __init__(
        self,
        scaling=1E4,
        dtype=np.uint8,
        *args,
        **kwargs,
    ):
        super(ALSDataset, self).__init__(*args, dtype=dtype, **kwargs)
        self.dataset_type = "ALS Timepix"
        self.scaling = scaling
        self.cache = self.read_data()
 
    def read_data(self):
        """
        read table of content for IMM datasets
        """
        x = np.loadtxt(self.fname, dtype=np.int64, skiprows=1)
        x = (x - np.min(x)) // int(self.scaling)
        print(np.min(x), np.max(x), x.dtype)
        data = np.bincount(x).reshape(-1, 1)
        data = data.astype(np.float32)
        print(data.shape, np.min(data), np.max(data), data[0])
        self.update_batch_info(data.size)
        return torch.tensor(data)
 
    def __getitem__(self, index):
        beg, end = self.get_raw_index(index)
        return self.cache[beg: end]


def example(fname='190k_r1.csv'):
    fname = os.path.join("/local/data_miaoqi/als_data", fname)

    batch_size = 4096
    scaling = 128 
    dset = ALSDataset(fname=fname, batch_size=batch_size,
                      det_size=(1, 1),
                      scaling=scaling,
                      device='cpu',
                      use_loader=False,
                      dtype=np.uint8,
                      mask=None)

    logger.info(f'scaling is {scaling}')
    xb = XB(dset.det_size, frame_num=dset.frame_num, queue_size=batch_size,
            max_count=4096)
    logger.info('solver created')

    stime = time.perf_counter()
    # dl = DataLoader(dset, num_workers=1, pin_memory=False)
    for n in trange(len(dset)):
        xb.process(dset[n])

    xb.post_process()
    etime = time.perf_counter()
    logger.info('multi-tau finished in %.3f s' % (etime - stime))

    result = xb.get_results()
    tau = result['tau'] * (1e-8 * scaling)
    g2 = result['g2'][:, 0]
    # print(tau.shape)
    # print(g2.shape)
    # als_res = np.vstack([tau, g2])
    # np.savetxt('tau-g2.txt', als_res)
    fig, ax = plt.subplots(1, 1)

    ax.semilogx(tau, g2)
    ax.semilogx(tau, g2, 'o')
    ax.set_xlabel('Delay (s)')
    ax.set_ylabel('g2')
    ax.set_title(os.path.basename(fname))
    plt.savefig(os.path.basename(fname).replace('.csv', '.png'), dpi=600)
    plt.show()
    plt.close(fig)


def test_als_dataset():
    fname = "/local/data_miaoqi/als_data/190k_r1.csv"
    dset = ALSDataset(fname=fname, batch_size=4096,
                      det_size=(1, 1),
                      device='cpu',
                      use_loader=False,
                      dtype=np.uint8,
                      mask=None)
    print(len(dset))


def run_all():
    flist = [
        '190k_r1.csv',
        '205k_r1.csv',
        '200k_r1.csv',
        '208k_r1.csv']
    for f in flist:
        example(f)


if __name__ == '__main__':
    # run_all()
    example()
    # test_als_dataset()

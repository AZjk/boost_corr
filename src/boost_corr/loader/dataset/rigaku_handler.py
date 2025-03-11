import numpy as np
import logging
import torch
<<<<<<<< HEAD:src/boost_corr/loader/rigaku_handler.py
from torch.utils.data import Dataset
from boost_corr.loader.xpcs_dataset import XpcsDataset
from boost_corr.utils.help_functions import convert_sparse
========
from .xpcs_dataset import XpcsDataset
from .help_functions import convert_sparse
>>>>>>>> 8f447fd767a5c7c182539a09654c813ad8d4d873:src/boost_corr/loader/dataset/rigaku_handler.py


logger = logging.getLogger(__name__)


class RigakuDataset(XpcsDataset):
    """
    Parameters
    ----------
    filename: string
        path to .imm file
    """
    def __init__(self, *args, dtype=np.uint8, total_frames=None, **kwargs):
        super(RigakuDataset, self).__init__(*args, dtype=dtype, **kwargs)
        self.dataset_type = "Rigaku 64bit Binary"
        self.is_sparse = True
        self.dtype = np.uint8
        self.ifc, self.mem_addr = self.read_data(total_frames)
        self.ifc = self.to_device()

    def to_device(self):
        index = torch.tensor(self.ifc[0].astype(np.int32), device=self.device)
        frame = torch.tensor(self.ifc[1].astype(np.int32), device=self.device)
        count = torch.tensor(self.ifc[2].astype(np.uint8), device=self.device)
        return (index, frame, count)

    def read_data(self, total_frames=None):
        with open(self.fname, 'r') as f:
            a = np.fromfile(f, dtype=np.uint64)
            d = convert_sparse(a)
            index, frame, count = d[0], d[1], d[2]
            if total_frames is None:
                total_frames = frame[-1] + 1
            else:
                total_frames = max(total_frames, frame[-1] + 1)

        # fix Rigaku3m module index offset
        if np.min(index) >= 1024 * 1024:
            index -= 1024 * 1024

        # update frame_num and batch_num
        self.update_batch_info(total_frames)

        all_index = np.arange(0, self.frame_num_raw + 2)
        all_index[-1] = self.frame_num_raw
        # build an index map for all indexes
        beg = np.searchsorted(frame, all_index[:-1])
        end = np.searchsorted(frame, all_index[1:])
        mem_addr = [slice(a, b) for a, b in zip(beg, end)]
        return (index, frame, count), mem_addr

    def __getbatch__(self, idx):
        # frame begin and end
        beg, end, size = self.get_raw_index(idx)
        x = torch.zeros(size, self.pixel_num, dtype=torch.uint8,
                        device=self.device)
        if self.stride == 1:
            # the data is continuous in RAM; convert by batch
            sla, slb = self.mem_addr[beg], self.mem_addr[end]
            a, b = sla.start, slb.start
            x[self.ifc[1][a:b].long() - beg, self.ifc[0][a:b].long()] = \
                self.ifc[2][a:b]
        else:
            # the data is discrete in RAM; convert frame by frame
            for n, idx in enumerate(np.arange(beg, end, self.stride)):
                sl = self.mem_addr[idx]
                x[n, self.ifc[0][sl].long()] = self.ifc[2][sl]
        
        if self.mask_crop is not None:
            x = x[:, self.mask_crop]
        return x


def test():
    fname = ("/home/beams/MQICHU/ramdisk/O018_Silica_D100_att0_Rq0_00001/"
             "O018_Silica_D100_att0_Rq0_00001.bin")
    ds = RigakuDataset(fname)
    # for n in range(len(ds)):
    #     print(n, ds[n].shape)


if __name__ == '__main__':
    test()

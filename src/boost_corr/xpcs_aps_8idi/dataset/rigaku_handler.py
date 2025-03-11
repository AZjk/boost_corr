"""Module for handling Rigaku datasets in the xpcs_aps_8idi dataset.
This module provides functionality to process Rigaku 64bit binary data.
TODO: Add detailed documentation.
"""

import logging

import numpy as np
import torch

from boost_corr.xpcs_aps_8idi.dataset.help_functions import convert_sparse
from boost_corr.xpcs_aps_8idi.dataset.xpcs_dataset import XpcsDataset

logger = logging.getLogger(__name__)


class RigakuDataset(XpcsDataset):
    """
    Parameters
    ----------
    filename: string
        path to .imm file
    """

    def __init__(self, *args, dtype=np.uint8, total_frames=None, **kwargs):
        """Initialize the RigakuDataset.

        Parameters:
            *args: Additional positional arguments.
            dtype: Data type for reading the data.
            total_frames: Total number of frames (optional).
            **kwargs: Additional keyword arguments.
        """
        super(RigakuDataset, self).__init__(*args, dtype=dtype, **kwargs)
        self.dataset_type = "Rigaku 64bit Binary"
        self.is_sparse = True
        self.dtype = np.uint8
        self.ifc, self.mem_addr = self.read_data(total_frames)
        self.ifc = self.to_device()

    def to_device(self):
        """Transfer internal data to the specified device and return processed tensors.

        Returns:
            Tuple containing tensors for index, frame, and count.
        """
        index = torch.tensor(self.ifc[0].astype(np.int32), device=self.device)
        frame = torch.tensor(self.ifc[1].astype(np.int32), device=self.device)
        count = torch.tensor(self.ifc[2].astype(np.uint8), device=self.device)
        return (index, frame, count)

    def read_data(self, total_frames=None):
        """Read data from the file and return it.

        Parameters:
            total_frames: (Optional) Total number of frames to read.

        Returns:
            Tuple containing the data and other related information.
        """
        with open(self.fname, "r") as f:
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
        """Retrieve a batch of data from the file.

        Parameters:
            idx: Batch index.

        Returns:
            Batch data.
        """
        # frame begin and end
        beg, end, size = self.get_raw_index(idx)
        x = torch.zeros(size, self.pixel_num, dtype=torch.uint8, device=self.device)
        if self.stride == 1:
            # the data is continuous in RAM; convert by batch
            sla, slb = self.mem_addr[beg], self.mem_addr[end]
            a, b = sla.start, slb.start
            x[self.ifc[1][a:b].long() - beg, self.ifc[0][a:b].long()] = self.ifc[2][a:b]
        else:
            # the data is discrete in RAM; convert frame by frame
            for n, idx in enumerate(np.arange(beg, end, self.stride)):
                sl = self.mem_addr[idx]
                x[n, self.ifc[0][sl].long()] = self.ifc[2][sl]

        if self.mask_crop is not None:
            x = x[:, self.mask_crop]
        return x


def test():
    """Test function for the RigakuDataset functionality."""
    _ = RigakuDataset()


if __name__ == "__main__":
    test()

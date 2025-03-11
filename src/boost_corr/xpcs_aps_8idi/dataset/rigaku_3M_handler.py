"""Module for handling Rigaku 3M datasets in the xpcs_aps_8idi dataset.
This module provides functionality to process Rigaku 64bit binary data.
TODO: Add detailed documentation.
"""

import logging
import os

import numpy as np
import torch

# from .help_functions import get_number_of_frames_from_binfile
from boost_corr.xpcs_aps_8idi.dataset.help_functions import (
    get_number_of_frames_from_binfile,
)

# from .rigaku_handler import RigakuDataset
from boost_corr.xpcs_aps_8idi.dataset.rigaku_handler import RigakuDataset

# from .xpcs_dataset import XpcsDataset
from boost_corr.xpcs_aps_8idi.dataset.xpcs_dataset import XpcsDataset

logger = logging.getLogger(__name__)


class Rigaku3MDataset(XpcsDataset):
    """Initialize the Rigaku3MDataset.

    Parameters:
        *args: Additional positional arguments.
        dtype: Data type for reading the data.
        gap: Gap tuple.
        layout: Layout tuple.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """

    def __init__(self, *args, dtype=np.uint8, gap=(70, 52), layout=(3, 2), **kwargs):
        """Initialize the Rigaku3MDataset instance.

        Parameters:
            *args: Additional positional arguments.
            dtype: Data type for reading the data.
            gap (tuple): Gap tuple between modules.
            layout (tuple): Layout specification for arranging modules.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super(Rigaku3MDataset, self).__init__(*args, dtype=dtype, **kwargs)
        self.dataset_type = "Rigaku 64bit Binary"
        self.is_sparse = True
        self.dtype = np.uint8
        self.container = self.get_modules(dtype, **kwargs)
        self.gap = gap
        self.layout = layout
        self.update_info()

    def get_modules(self, dtype, **kwargs):
        """Get the module ordering.

        Parameters:
            dtype: Data type.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple representing the module order.
        """
        index_list = (5, 0, 4, 1, 3, 2)
        flist = [self.fname[:-3] + f"00{n}" for n in index_list]
        for f in flist:
            assert os.path.isfile(f)
        # determine the total number of frames
        total_frames = [get_number_of_frames_from_binfile(f) for f in flist]
        for n, f in enumerate(flist):
            logger.info(f"{f}: {total_frames[n]}")
        total_frames = max(total_frames)

        kwargs.pop("mask_crop", None)
        return [
            RigakuDataset(
                f, dtype=dtype, mask_crop=None, total_frames=total_frames, **kwargs
            )
            for f in flist
        ]

    def update_info(self):
        """Update dataset information.

        Raises:
            AssertionError: If frame numbers mismatch among modules.
        """
        frame_num = [x.frame_num for x in self.container]
        assert len(set(frame_num)) == 1, "frame number mismatch in the 6 modules"
        self.frame_num = frame_num[0]
        shape_one = self.container[0].det_size
        shape = [
            shape_one[n] * self.layout[n] + self.gap[n] * (self.layout[n] - 1)
            for n in range(2)
        ]
        self.det_size = tuple(shape)
        self.det_size_one = shape_one

    def append_data(self, canvas, index, data_module):
        """Append data from a module to the canvas.

        Parameters:
            canvas: The canvas to append to.
            index: Index indicating position.
            data_module: Data from the module.

        Returns:
            The updated canvas.
        """
        row = index // 2
        col = index % 2
        st_v = row * (self.det_size_one[0] + self.gap[0])
        sl_v = slice(st_v, st_v + self.det_size_one[0])
        st_h = col * (self.det_size_one[1] + self.gap[1])
        sl_h = slice(st_h, st_h + self.det_size_one[1])
        canvas[:, sl_v, sl_h] = data_module.reshape(-1, *self.det_size_one)

        return canvas

    def __getbatch__(self, idx):
        """Retrieve a batch of data.

        Parameters:
            idx: Batch index.

        Returns:
            Batch data.
        """
        _, _, size = self.get_raw_index(idx)
        canvas = torch.zeros(
            size, *self.det_size, dtype=torch.uint8, device=self.device
        )
        for index, module in enumerate(self.container):
            temp = module.__getbatch__(idx)
            canvas = self.append_data(canvas, index, temp)

        canvas = canvas.reshape(size, -1)
        if self.mask_crop is not None:
            canvas = canvas[:, self.mask_crop]

        return canvas


def test():
    """Test function for Rigaku3MDataset functionality."""
    fname = "../../../tests/data/verify_circular_correlation/F091_D100_Capillary_Post_att00_Lq0_Rq0_00001/F091_D100_Capillary_Post_att00_Lq0_Rq0_00001.bin"
    _ = RigakuDataset(fname)


if __name__ == "__main__":
    test()

import numpy as np
import logging
import torch
from .xpcs_dataset import XpcsDataset
from timepix_dataset.dataset import create_timepix_dataset


logger = logging.getLogger(__name__)


class TimepixAnalysisDataset(XpcsDataset):
    def __init__(
        self,
        inner_dataset,
        *args,
        dtype=np.uint8,
        total_frames=None,
        bin_time_s=1e-6,
        **kwargs,
    ):
        super(TimepixAnalysisDataset, self).__init__(*args, dtype=dtype, **kwargs)
        self.dataset_type = "Timepix4Dataset"
        self.is_sparse = True
        self.dtype = np.uint8
        self.inner_dataset = inner_dataset
        self.update_det_size(self.inner_dataset.det_size)

        self.ifc, self.mem_addr = self.read_data(total_frames, bin_time_s=bin_time_s)
        self.ifc = self.to_device()

    def to_device(self, max_fsize_gb_in_gpu=1.0):
        total_size = sum([arr.nbytes for arr in self.ifc]) / 1024**3
        if total_size <= max_fsize_gb_in_gpu and self.device.startswith("cuda"):
            device = self.device
        else:
            device = "cpu"
            logger.info(f"Total data size {total_size:.2f} GB, offloading to CPU RAM")
        index = torch.tensor(self.ifc[0], device=device)
        frame = torch.tensor(self.ifc[1], device=device)
        count = torch.tensor(self.ifc[2], device=device)
        return (index, frame, count)

    def read_data(self, total_frames=None, bin_time_s=1e-6):
        ifc = self.inner_dataset.apply_time_binning(bin_time_s)
        index, frame, count = ifc
        # update frame_num and batch_num
        self.update_batch_info(frame[-1] + 1)

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
        device = self.ifc[0].device
        x = torch.zeros(size, self.pixel_num, dtype=torch.uint8, device=device)
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
        x = x.to(self.device)
        if self.mask_crop is not None:
            x = x[:, self.mask_crop]
        return x

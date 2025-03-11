import numpy as np

# required for bloc compression
import hdf5plugin
import h5py
import torch
import logging
from .xpcs_dataset import XpcsDataset


logger = logging.getLogger(__name__)


class HdfDataset(XpcsDataset):
    """
    Parameters
    ----------
    filename: string
    """
    def __init__(self,
                 *args,
                 preload_size=8,
                 dtype=np.uint8,
                 data_path='/entry/data/data',
                 **kwargs):

        super(HdfDataset, self).__init__(*args, dtype=dtype, **kwargs)
        self.dataset_type = "HDF5 Dataset"
        self.is_sparse = False
        with h5py.File(self.fname, 'r') as f:
            data = f[data_path]
            self.shape = data.shape

            # update data type;
            if data.dtype == np.uint8:
                self.dtype = np.int16
            elif data.dtype == np.uint16:
                # lambda2m's uint16 is actually 12bit. it's safe to to int16
                if self.shape[1] * self.shape[2] == 1813 * 1558:
                    self.dtype = np.int16
                # likely eiger detectors
                else:
                    self.dtype = np.int32
            elif data.dtype == np.uint32:
                self.dtype = np.int32
                logger.warn('cast uint32 to int32. it may cause ' + 
                            'overflow when the maximal value >= 2^31')
            else:
                self.dtype = data.dtype
        
        if kwargs['avg_frame'] > 1:
            self.dtype = np.float32
        self.fhdl = None
        self.data = None
        self.data_cache = None
        self.data_path = data_path

        self.update_batch_info(self.shape[0])
        self.update_det_size(self.shape[1:])

        if self.mask_crop is not None:
            self.mask_crop = self.mask_crop.cpu().numpy()

        self.current_group = None
        self.preload_size = preload_size
        self.cache = None

    def __reset__(self):
        if self.fhdl is not None:
            self.fhdl.close()
            self.data = None
            self.fhdl = None

    def __getbatch__(self, idx):
        """
        return numpy array, which will be converted to tensor with dataloader
        """
        if self.fhdl is None:
            self.fhdl = h5py.File(self.fname, 'r', rdcc_nbytes=1024*1024*256)
            self.data = self.fhdl[self.data_path]

        beg, end, size = self.get_raw_index(idx)
        if self.cache is None or self.cache.shape[0] != size:
            self.cache = np.zeros(shape=(size, self.shape[1], self.shape[2]), dtype=self.dtype)
        # idx_list = np.arange(beg, end, self.stride)

        self.data.read_direct(self.cache, np.s_[beg:end:self.stride])
        x = self.cache.reshape(size, self.pixel_num)
        if self.mask_crop is not None:
            x = x[:, self.mask_crop]

        return torch.from_numpy(x)


def test():
    fname = (
        "/clhome/MQICHU/ssd/xpcs_data_raw/A003_Cu3Au_att0_001/A003_Cu3Au_att0_001.imm"
    )
    ds = HdfDataset(fname)
    # for n in range(len(ds)):
    #     print(n, ds[n].shape, ds[n].device)


def test_bin(idx):
    fname = f"/clhome/MQICHU/ssd/APSU_TestData_202106/APSU_TestData_{idx:03d}/APSU_TestData_{idx:03d}.h5"
    ds = HdfDataset(fname)
    ds.to_rigaku_bin(f"hdf2bin_{idx:03d}.bin")


if __name__ == '__main__':
    test_bin(4)

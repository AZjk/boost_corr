import numpy as np

# required for bloc compression
import hdf5plugin
import h5py
import torch
from .xpcs_dataset import XpcsDataset


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
            # lambda2m's uint16 is actually 12bit
            if data.dtype in [np.uint8, np.uint16]:
                self.dtype = np.int16
            else:
                self.dtype = data.dtype

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
            self.fhdl = h5py.File(self.fname, 'r')
            self.data = self.fhdl[self.data_path]

        beg, end, size = self.get_raw_index(idx)
        idx_list = np.arange(beg, end, self.stride)

        if self.mask_crop is not None:
            # x = self.data[beg:end, self.sl_v, self.sl_h].reshape(end - beg, -1)
            x = self.data[idx_list].reshape(size, -1)
            x = x[:, self.mask_crop].astype(self.dtype)
            # if idx == 0:
            #     print(np.max(x), x.dtype)
        else:
            x = self.data[idx_list].reshape(-1, self.pixel_num)

        if x.dtype != self.dtype:
            x = x.astype(self.dtype)
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

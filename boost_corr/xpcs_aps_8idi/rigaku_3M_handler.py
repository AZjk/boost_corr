import numpy as np
from .xpcs_dataset import XpcsDataset
from .rigaku_handler import RigakuDataset
import os
import logging
import torch


logger = logging.getLogger(__name__)


class Rigaku3MDataset(XpcsDataset):
    """
    Parameters
    ----------
    filename: string
        path to Rigaku3M binary.000 file
    """
    def __init__(self, *args, dtype=np.uint8, gap=(71, 50), layout=(3, 2),
                 **kwargs):
        super(Rigaku3MDataset, self).__init__(*args, dtype=dtype, **kwargs)
        self.dataset_type = "Rigaku 64bit Binary"
        self.is_sparse = True
        self.dtype = np.uint8
        self.container = self.get_modules(dtype, **kwargs)
        self.gap = gap
        self.layout = layout
        self.update_info()
    
    def get_modules(self, dtype, **kwargs):
        # this is the order for the 6 modules
        index_list = (5, 0, 4, 1, 3, 2)
        flist = [self.fname[:-3] + f'00{n}' for n in index_list]
        for f in flist:
            assert os.path.isfile(f)
        return [RigakuDataset(f, dtype=dtype, **kwargs) for f in flist]
    
    def update_info(self):
        frame_num = [x.frame_num for x in self.container]
        assert len(list(set(frame_num))) == 1, 'check frame numbers'
        self.update_batch_info(frame_num[0])
        shape_one = self.container[0].det_size 
        shape = [shape_one[n] * self.layout[n] + self.gap[n] * (self.layout[n] - 1) for n in range(2)]
        self.det_size = tuple(shape)
        self.det_size_one = shape_one
    
    def patch_data(self, scat_list):
        gap = self.gap
        layout = self.layout

        shape_one = scat_list[0].shape
        shape = self.det_size

        canvas = np.zeros(shape, dtype=scat_list[0].dtype)
        #  canvas[:, :] = np.nan
        for row in range(layout[0]):
            st_v = row * (shape_one[0] + gap[0])
            sl_v = slice(st_v, st_v + shape_one[0])
            for col in range(layout[1]):
                st_h = col * (shape_one[1] + gap[1])
                sl_h = slice(st_h, st_h + shape_one[1])
                canvas[sl_v, sl_h] = scat_list[row * layout[1] + col]
        return canvas
    
    def append_data(self, canvas, index, data_module):
        shape_one = (512, 1024)

        row = index // 2
        col = index % 2
        st_v = row * (shape_one[0] + self.gap[0])
        sl_v = slice(st_v, st_v + shape_one[0])
        st_h = col * (shape_one[1] + self.gap[1])
        sl_h = slice(st_h, st_h + shape_one[1])
        canvas[:, sl_v, sl_h] = data_module.reshape(-1, *self.det_size_one)

        return canvas
    
    def __getbatch__(self, idx):
        # frame begin and end
        beg, end, size = self.get_raw_index(idx)
        canvas = torch.zeros(size, *self.det_size, dtype=torch.uint8,
                             device=self.device)
        for index, module in enumerate(self.container):
            temp = module.__getbatch__(idx)
            canvas = self.append_data(canvas, index, temp)

        return canvas.reshape(size, -1)


def test():
    fname = "../../../tests/data/verify_circular_correlation/F091_D100_Capillary_Post_att00_Lq0_Rq0_00001/F091_D100_Capillary_Post_att00_Lq0_Rq0_00001.bin"
    ds = RigakuDataset(fname)


if __name__ == '__main__':
    test()

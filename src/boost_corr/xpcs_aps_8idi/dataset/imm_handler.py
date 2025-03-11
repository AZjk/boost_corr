import struct
import numpy as np
import time
import logging
import os
from tqdm import trange
import torch
from .xpcs_dataset import XpcsDataset

logger = logging.getLogger(__name__)


def read_imm_header(file):
    imm_headformat = "ii32s16si16siiiiiiiiiiiiiddiiIiiI40sf40sf40sf40s" + \
                     "f40sf40sf40sf40sf40sf40sfffiiifc295s84s12s"
    imm_fieldnames = [
        'mode', 'compression', 'date', 'prefix', 'number', 'suffix', 'monitor',
        'shutter', 'row_beg', 'row_end', 'col_beg', 'col_end', 'row_bin',
        'col_bin', 'rows', 'cols', 'bytes', 'kinetics', 'kinwinsize',
        'elapsed', 'preset', 'topup', 'inject', 'dlen', 'roi_number',
        'buffer_number', 'systick', 'pv1', 'pv1VAL', 'pv2', 'pv2VAL', 'pv3',
        'pv3VAL', 'pv4', 'pv4VAL', 'pv5', 'pv5VAL', 'pv6', 'pv6VAL', 'pv7',
        'pv7VAL', 'pv8', 'pv8VAL', 'pv9', 'pv9VAL', 'pv10', 'pv10VAL',
        'imageserver', 'CPUspeed', 'immversion', 'corecotick', 'cameratype',
        'threshhold', 'byte632', 'empty_space', 'ZZZZ', 'FFFF'
    ]

    bindata = file.read(1024)

    imm_headerdat = struct.unpack(imm_headformat, bindata)
    imm_header = dict(zip(imm_fieldnames, imm_headerdat))

    return imm_header


class ImmDataset(XpcsDataset):
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
        *args,
        dtype=np.int16,
        frames_per_point=1,
        **kwargs,
    ):
        super(ImmDataset, self).__init__(*args, dtype=dtype, **kwargs)
        self.dataset_type = "IMM Legacy"
        self.frames_per_point = frames_per_point

        self.toc, self.det_size = self.read_toc()

        # update frame_num and batch info
        self.update_batch_info(self.toc.shape[0])
        self.update_det_size(self.det_size)
        self.fh = None

    def read_toc(self):
        """
        read table of content for IMM datasets
        """
        with open(self.fname, "rb") as f:
            header = read_imm_header(f)
            det_size = (header['rows'], header['cols'])
            self.is_sparse = bool(header['compression'] == 6)
            f.seek(0)
            toc = []  # (start byte, element count) pairs
            while True:
                try:
                    header = read_imm_header(f)
                    cur = f.tell()
                    payload_size = header['dlen'] * \
                        (6 if self.is_sparse else 2)
                    toc.append((cur, header['dlen']))

                    file_pos = payload_size + cur
                    f.seek(file_pos)

                    # Check for end of file.
                    if not f.peek(4):
                        break

                except Exception as err:
                    raise IOError("IMM file is corrupted.")

            return np.array(toc), det_size

    def __reset__(self):
        self.fh = None

    def __getbatch__(self, index):
        if self.fh is None:
            self.fh = open(self.fname, 'rb')

        if self.is_sparse:
            x = self.__get_frame_sparse__(index)
        else:
            x = self.__get_frame_dense__(index)

        if self.mask_crop is not None:
            x = x[:, self.mask_crop]
        return x

    def __get_frame_dense__(self, batch_idx):
        beg, end, size = self.get_raw_index(batch_idx)
        idx_list = np.arange(beg, end, self.stride)
        toc = self.toc[idx_list]
        imgs = []
        for (start_byte, event_num) in toc:
            self.fh.seek(start_byte)
            imgs.append(np.fromfile(self.fh, dtype=np.uint16, count=event_num))
        imgs = np.array(imgs).astype(np.int16)
        x = torch.from_numpy(imgs).to(self.device, non_blocking=True)
        return x

    def __get_frame_sparse__(self, batch_idx):
        beg, end, size = self.get_raw_index(batch_idx)
        idx_list = np.arange(beg, end, self.stride)
        toc = self.toc[idx_list]

        frame = []
        index = []
        count = []
        for n, (start_byte, event_num) in enumerate(toc):
            self.fh.seek(start_byte)
            frame.append(np.zeros(event_num, dtype=np.int16) + n)
            index.append(np.fromfile(self.fh, dtype=np.int32, count=event_num))
            count.append(
                np.fromfile(self.fh, dtype=self.dtype, count=event_num))

        # frame = torch.tensor(np.concatenate(frame), device=self.device)
        frame = torch.from_numpy(np.concatenate(frame)).to(self.device,
                non_blocking=True)
        index = torch.from_numpy(np.concatenate(index)).to(self.device,
                non_blocking=True)
        count = torch.from_numpy(np.concatenate(count)).to(self.device,
                non_blocking=True)

        return self.sparse_to_dense(index, frame, count, size)

    def __del__(self):
        try:
            self.fh.close()
        except Exception:
            pass


def read_data(file_name, batch_size=1024):
    logger.info('start, fname is: %s', os.path.basename(file_name))
    logger.info('dirname is: %s', os.path.dirname(file_name))
    imm = ImmDataset(file_name, batch_size=1024)
    # print(imm.toc[0])
    # print(imm.toc[1])
    # print(len(imm))
    # print(imm[1])
    # print(imm.is_sparse)
    # print(imm.det_size)
    logger.info('toc is generated')
    logger.info('toc length is %d', len(imm.toc))
    tot = 0
    stime = time.perf_counter()
    for n in range(len(imm)):
        x = imm[n]
        # tot += torch.sum(x)
    etime = time.perf_counter()
    logger.info('sum of array is %d', tot)
    t_diff = (etime - stime)
    freq = imm.frame_num / t_diff
    print(f"data traversed: {t_diff:02f}s / {freq:04f}Hz")
    return freq


def test01():
    file_name = "/scratch/xpcs_data_raw/" + \
                "A005_Dragonite_25p_Quiescent_att0_Lq0_001/" + \
                "A005_Dragonite_25p_Quiescent_att0_Lq0_001_00001-20000.imm"
    read_data(file_name)


def test02():
    fname = "/home/8ididata/2021-3/hallinan202111/W2397_S15-SEO-C-S_Lq1_080C_att06_001/W2397_S15-SEO-C-S_Lq1_080C_att06_001_00001-01000.imm"
    read_data(fname)


def test03():
    file_name = "/scratch/xpcs_data_raw/" + \
                "A005_Dragonite_25p_Quiescent_att0_Lq0_001/" + \
                "A005_Dragonite_25p_Quiescent_att0_Lq0_001_00001-20000.imm"
    for batch_size in (2 ** np.arange(12)):
        print(batch_size, read_data(file_name, batch_size))


if __name__ == '__main__':
    test03()

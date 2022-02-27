import numpy as np
import os
import h5py
import logging
from Append_Metadata_xpcs_multitau import append_qmap
import glob2
from hdf_reader import put, get
from xpcs_qpartitionmap import XpcsQPartitionMap

logger = logging.getLogger(__name__)


def is_metadata(fname: str):
    with h5py.File(fname, "r") as f:
        if "/hdf_metadata_version" in f:
            return True
        else:
            return False


def get_metadata(meta_dir: str):
    meta_fname = glob2.glob(meta_dir + "/*.hdf")
    for x in meta_fname:
        if is_metadata(x):
            return x
    raise FileExistsError(f'no metadata file found in [{meta_dir}]')


class XpcsResult(object):
    def __init__(self,
                 meta_dir: str,
                 qmap_fname: str,
                 output_dir: str,
                 overwrite=False,
                 analysis_type='Multitau') -> None:
        super().__init__()

        meta_fname = get_metadata(meta_dir)
        # the filename for the result file;
        fname = os.path.join(output_dir, os.path.basename(meta_fname))
        if analysis_type == 'Twotime':
            fname = fname.replace('.hdf', '_Twotime.hdf')
        # avoid overwrite
        idx = 1
        while not overwrite and os.path.isfile(fname):
            if idx == 1:
                fname = fname[:-4] + f'_{idx:02d}.hdf'
            else:
                fname = fname[:-7] + f'_{idx:02d}.hdf'
            idx += 1
        self.fname = fname

        append_qmap(meta_fname, qmap_fname, fname,
                    analysis_type=analysis_type)

    def save(self, result_dict, mode="alias", compression=None, **kwargs):
        put(self.fname, result_dict, mode=mode, compression=compression,
            **kwargs)


def test_01():
    dir = "../O018_Silica_D100_att0_Rq0_00005"
    fname = "O018_Silica_D100_att0_Rq0_00005_0001-100000.hdf"
    path = os.path.join(dir, fname)

    detector = XpcsResult(path)
    detector.show()


def test_02():
    meta_dir = "/home/miaoqi/Work/xpcs_data_raw/A005_Dragonite_25p_Quiescent_att0_Lq0_001"
    qmap_fname = "/home/miaoqi/Work/xpcs_data_raw/qmap/harden201912_qmap_Dragonite_Lq0_S270_D54.h5"
    output_dir = "/home/miaoqi/Work/xpcs_data_raw/cluster_result"
    a = XpcsResult(meta_dir, qmap_fname, output_dir)


if __name__ == "__main__":
    test_02()

import os
import h5py
import logging
import glob2
import traceback
from .Append_Metadata_xpcs_multitau import append_qmap
from .hdf_reader import put

logger = logging.getLogger(__name__)


def is_metadata(fname: str):
    if not os.path.isfile(fname):
        return False

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
    raise FileNotFoundError(f'no metadata file found in [{meta_dir}]')


class XpcsResult(object):
    def __init__(self,
                 meta_dir: str,
                 qmap_fname: str,
                 output_dir: str,
                 fname=None,
                 overwrite=False,
                 avg_frame=1,
                 stride_frame=1,
                 analysis_type='Multitau') -> None:
        super().__init__()

        meta_fname = get_metadata(meta_dir)
        # the filename for the result file;
        if fname is None:
            fname = os.path.join(output_dir, os.path.basename(meta_fname))
        else:
            # for pvapy applications
            fname = fname.replace(':', '_')
            if not fname.endswith('.hdf'):
                fname += '.hdf'
            fname = os.path.join(output_dir, fname)

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

        try:
            append_qmap(meta_fname, qmap_fname, fname, avg_frame=avg_frame,
                        stride_frame=stride_frame, analysis_type=analysis_type)
        except Exception:
            traceback.print_exc()
            raise IOError(f'Check metadata file {meta_fname}')
        
    def save(self, result_dict, mode="alias", compression=None, **kwargs):
        if 'saxs_2d_single' in result_dict:
            result_dict.pop('saxs_2d_single')
        put(self.fname, result_dict, mode=mode, compression=compression,
            **kwargs)
        logger.info(f'correlation result saved to [{self.fname}]')


if __name__ == "__main__":
    pass

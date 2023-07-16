import os
import h5py
import logging
import glob
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
    meta_fname = glob.glob(meta_dir + "/*.hdf")
    for x in meta_fname:
        if is_metadata(x):
            return x
    raise FileNotFoundError(f'no metadata file found in [{meta_dir}]')


class XpcsResult(object):
    def __init__(self,
                 meta_dir: str,
                 qmap_fname: str,
                 output_dir: str,
                 overwrite=False,
                 avg_frame=1,
                 stride_frame=1,
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

        try:
            append_qmap(meta_fname, qmap_fname, fname, avg_frame=avg_frame,
                        stride_frame=stride_frame, analysis_type=analysis_type)
        except Exception:
            traceback.print_exc()
            raise IOError(f'Check metadata file {meta_fname}')

    def save(self, result_dict, mode="alias", compression=None, **kwargs):
        if 'G2IPIF' not in result_dict:
            put(self.fname, result_dict, mode=mode, compression=compression,
                **kwargs)
        else:
            # save G2 etc
            G2 = result_dict.pop('G2IPIF')
            G2_fname = os.path.splitext(self.fname)[0] + '_G2.hdf'
            put(G2_fname, {'G2IPIF': G2}, mode="raw",
                compression=compression, **kwargs)
            put(self.fname, result_dict, mode=mode, compression=compression,
                **kwargs)
            # create a link for the main file
            with h5py.File(self.fname, 'r+') as f:
                relative_path = os.path.basename(G2_fname)
                f['/exchange/G2IPIF'] = h5py.ExternalLink(relative_path, 
                                                          "/G2IPIF")


if __name__ == "__main__":
    pass

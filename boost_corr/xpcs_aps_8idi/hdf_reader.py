import h5py
import numpy as np
import logging


logger = logging.getLogger(__name__)

from .aps_8idi import key as hdf_key


def put_results_in_hdf5(save_path, result, mode='raw', compression=None):
    """
    save the results to a hdf5 file;
    Parameters
    ----------
    save_path: str
        path to save the hdf5 file
    result: dict
        dictionary of results to save
    mode: str
        ['raw' | 'alias']
    compression: str
        compression method to use
    """
    with h5py.File(save_path, 'a') as f:
        for key, val in result.items():
            if mode == 'alias':
                key = hdf_key[key]
            if key in f:
                del f[key]
            if isinstance(val, dict) and len(val) > 0:
                group = f.create_group(key)
                for k2, v2 in val.items():
                    group.create_dataset(k2, data=v2)
            else:
                if isinstance(val, np.ndarray) and val.size > 1024 and compression is None:
                    f.create_dataset(key, data=val, compression='gzip',
                                     compression_opts=4, chunks=True)
                else:
                    f.create_dataset(key, data=val, compression=compression)
        return




if __name__ == '__main__':
    pass

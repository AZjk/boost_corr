import h5py
import numpy as np
import logging
import traceback


logger = logging.getLogger(__name__)

from .aps_8idi import key as hdf_key


def put_results_in_hdf5(save_path, result, compression=None):
    """
    Save results to an HDF5 file with optional compression.

    Parameters
    ----------
    save_path : str
        Path to save the HDF5 file
    result : dict
        Dictionary of results to save
    compression : str, optional
        Compression method to use. If None, large arrays use gzip compression

    Returns
    -------
    None
    """
    def create_dataset_with_compression(f, path, data):
        """Helper function to create dataset with appropriate compression"""
        if isinstance(data, np.ndarray) and data.size > 8192 and compression is None:
            # chunks using the last two dimensions
            if data.ndim >= 2:
                    chunks = tuple([1] * (data.ndim - 2) + list(data.shape[-2:]))
            else:
                chunks = data.shape
            return f.create_dataset(path, data=data, 
                                    compression='lzf',
                                    chunks=chunks)
        return f.create_dataset(path, data=data, compression=compression)

    def save_dict_to_group(group, dictionary):
        """Helper function to save dictionary to HDF5 group"""
        for key, value in dictionary.items():
            if value is not None:
                group.create_dataset(key, data=value)

    try:
        with h5py.File(save_path, 'a') as f:
            for key, value in result.items():
                # Skip None values
                if value is None:
                    continue

                # Determine path
                path = (hdf_key["c2_prefix"] + '/' + key 
                       if key.startswith('correlation_map/') 
                       else hdf_key[key])

                # Delete existing dataset/group if it exists
                if path in f:
                    del f[path]

                # Handle dictionary values
                if isinstance(value, dict):
                    if value:  # Check if dictionary is not empty
                        group = f.create_group(path)
                        save_dict_to_group(group, value)
                # Handle non-dictionary values
                else:
                    create_dataset_with_compression(f, path, value)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
        raise


if __name__ == '__main__':
    pass

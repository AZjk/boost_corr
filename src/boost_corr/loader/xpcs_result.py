import os
import sys
import h5py
import logging
import glob
import traceback
import numpy
from boost_corr.loader.Append_Metadata_xpcs_multitau import append_qmap
from boost_corr.loader.hdf_reader import put
from typing import Any

logger = logging.getLogger(__name__)




def isHdf5FileObject(obj):
    """Is `obj` an HDF5 File?"""
    return isinstance(obj, h5py.File)


def isHdf5Group(obj):
    """Is `obj` an HDF5 Group?"""
    return isinstance(obj, h5py.Group) and not isHdf5FileObject(obj)


def decode_byte_string(value):
    """Convert (arrays of) byte-strings to (list of) unicode strings.

    Due to limitations of HDF5, all strings are saved as byte-strings or arrays
    of byte-stings, so they must be converted back to unicode. All other typed
    objects pass unchanged.

    Zero-dimenstional arrays are replaced with None.
    """
    if (isinstance(value, numpy.ndarray) and value.dtype.kind in ['O', 'S']):
        if value.size > 0:
            return value.astype('U').tolist()
        else:
            return None
    elif isinstance(value, (bytes, numpy.bytes_)):
        return value.decode(sys.stdout.encoding or "utf8")
    else:
        return value


def isNeXusGroup(obj, NXtype):
    """Is `obj` a NeXus group?"""
    nxclass = None
    if isHdf5Group(obj):
        nxclass = obj.attrs.get("NX_class", None)
        if isinstance(nxclass, numpy.ndarray):
            nxclass = nxclass[0]
        nxclass = decode_byte_string(nxclass)
    return nxclass == str(NXtype)


def isNeXusFile(filename):
    """Is `filename` is a NeXus HDF5 file?"""
    if not os.path.exists(filename):
        return None

    with h5py.File(filename, "r") as root:
        if isHdf5FileObject(root):
            for item in root:
                try:
                    if isNeXusGroup(root[item], "NXentry"):
                        return True
                except KeyError:
                    pass
    return False


def is_metadata(fname: str) -> bool:
    """
    Determine if the specified file contains metadata by checking for the
    presence of the 'hdf_metadata_version' dataset.

    Parameters:
        fname (str): The path to the file.

    Returns:
        bool: True if the file contains metadata; False otherwise.
    """
    if not fname or not os.path.isfile(fname):
        return False
    try:
        with h5py.File(fname, 'r') as f:
            return 'hdf_metadata_version' in f
    except Exception:
        return False


def get_metadata(directory: str) -> str:
    """
    Locate the metadata file within the specified directory.

    This function searches for .hdf files in the directory that contain metadata,
    determined by the is_metadata function.

    Parameters:
        directory (str): The directory to search.

    Returns:
        str: The path to the metadata file.

    Raises:
        FileNotFoundError: If no metadata file is found in the directory.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    for fname in os.listdir(directory):
        if fname.endswith('.hdf'):
            full_path = os.path.join(directory, fname)
            if is_metadata(full_path):
                return full_path
    raise FileNotFoundError("Metadata file not found in the directory.")


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

        meta_fname, meta_ftype = get_metadata(meta_dir)
        logger.info(f'metadata file type is {meta_ftype=}')
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
            append_qmap(meta_ftype, meta_fname, qmap_fname, fname, 
                        avg_frame=avg_frame, stride_frame=stride_frame,
                        analysis_type=analysis_type)
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

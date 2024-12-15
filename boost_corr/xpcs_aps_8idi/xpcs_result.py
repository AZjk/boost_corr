import glob
import logging
import os
import shutil
import sys
import traceback
from typing import Optional

import h5py
import numpy

from .Append_Metadata_xpcs_multitau import append_qmap
from .hdf_reader import put

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


def is_metadata(fname: str):
    if not os.path.isfile(fname):
        return False, None

    with h5py.File(fname, "r") as f:
        if "/hdf_metadata_version" in f:
            return True, 'legacy'
        elif isNeXusFile(fname):
            return True, 'nexus'
        else:
            return False, None


def get_metadata(meta_dir: str):
    meta_fname = glob.glob(meta_dir + "/*.hdf")
    for x in meta_fname:
        flag, ftype = is_metadata(x)
        if flag:
            return x, ftype
    raise FileNotFoundError(f'no metadata file found in [{meta_dir}]')


def create_unique_file(
    output_dir: str,
    meta_fname: str,
    analysis_type: Optional[str] = None,
    overwrite: bool = False
) -> str:
    """
    Create a unique filename for an HDF file, avoiding overwriting existing files.

    This function generates a unique filename based on the input parameters. It ensures
    the file has an .hdf extension, incorporates the analysis type if specified, and
    avoids overwriting existing files by appending a number to the filename if necessary.

    Args:
        output_dir (str): The directory where the file will be created.
        meta_fname (str): The base filename to use.
        analysis_type (Optional[str], optional): The type of analysis, if 'Twotime' is
            specified, '_Twotime' will be added to the filename. Defaults to None.
        overwrite (bool, optional): If True, allows overwriting existing files.
            Defaults to False.

    Returns:
        str: A unique filename (including path) that can be used to create a new file
             without overwriting existing files, unless overwrite is True.

    Examples:
        >>> create_unique_file('/output', 'data.txt')
        '/output/data.hdf'
        >>> create_unique_file('/output', 'data.hdf', analysis_type='Twotime')
        '/output/data_Twotime.hdf'
        >>> create_unique_file('/output', 'data.hdf', overwrite=False)
        '/output/data_01.hdf'  # if 'data.hdf' already exists
    """
    # Create the base filename
    base_fname = os.path.basename(meta_fname)
    
    # Ensure the filename has the .hdf extension
    name, ext = os.path.splitext(base_fname)
    if ext.lower() != '.hdf':
        base_fname = name + '.hdf'

    # Add analysis type to the filename if specified
    if analysis_type == 'Twotime':
        base_fname = base_fname.replace('.hdf', '_Twotime.hdf')

    # Create the full path
    fname = os.path.join(output_dir, base_fname)

    # If overwrite is True, return the filename as is
    if overwrite or not os.path.isfile(fname):
        return fname

    # If the file exists and overwrite is False, create a unique filename
    name, ext = os.path.splitext(fname)
    counter: int = 1
    while True:
        new_fname = f"{name}_{counter}{ext}"
        if not os.path.isfile(new_fname):
            return new_fname
        counter += 1


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
        logger.info(f'metadata filename/type is {meta_fname} | {meta_ftype}')
        self.fname = create_unique_file(output_dir, meta_fname,
                                        analysis_type=analysis_type,
                                        overwrite=overwrite)
        self.G2_fname = os.path.splitext(self.fname)[0] + '_G2.hdf'
        self.fname_temp = self.fname + '.temp'
        self.G2_fname_temp = self.G2_fname + '.temp'
        self.result_kwargs = {
            'meta_fname': meta_fname,
            'qmap_fname': qmap_fname,
            'meta_type': meta_ftype,
            'avg_frame': avg_frame,
            'stride_frame': stride_frame,
            'analysis_type': analysis_type
        }

    def save_as_temp(self, result_dict, mode="alias", compression=None,
                     **kwargs):
        append_qmap(self.fname_temp, **self.result_kwargs)
        G2 = result_dict.pop('G2IPIF', None)
        put(self.fname_temp, result_dict, mode=mode, compression=compression,
            **kwargs)
        if G2 is not None:
            put(self.G2_fname_temp, {'G2IPIF': G2}, mode="raw",
                compression=compression, **kwargs)
            # create a link for the main file
            with h5py.File(self.fname_temp, 'r+') as f:
                relative_path = os.path.basename(self.G2_fname)
                f['/exchange/G2IPIF'] = h5py.ExternalLink(relative_path, 
                                                          "/G2IPIF")

    def safe_save(self, *args, **kwargs):
        flag = False
        try:
            self.save_as_temp(*args, **kwargs)
            if os.path.isfile(self.fname_temp):
                shutil.move(self.fname_temp, self.fname)
            if os.path.isfile(self.G2_fname_temp):
                shutil.move(self.G2_fname_temp, self.G2_fname)
            flag = True
        except FileNotFoundError as fnf_error:
            logger.error(f"File not found: {fnf_error}")
        except PermissionError as perm_error:
            logger.error(f"Permission denied: {perm_error}")
        except OSError as os_error:
            logger.error(f"OS error occurred: {os_error}")
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")
            traceback.print_exc()
        finally:
            # Clean up temporary files if they still exist
            for temp_file in [self.G2_fname_temp, self.fname_temp]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        logger.info(f"Cleaned up temporary file: {temp_file}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temporary file {temp_file}: {cleanup_error}")
        if flag:
            logger.info(f"Succeeded in saving result file: {self.fname}")
        else:
            logger.error(f"Failed in saving result file: {self.fname}")
        return flag

if __name__ == "__main__":
    pass

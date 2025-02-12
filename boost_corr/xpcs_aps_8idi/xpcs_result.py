import glob
import logging
import os
import shutil
import traceback
from typing import Optional

import h5py

from .Append_Metadata_xpcs_multitau import append_qmap
from .hdf_reader import put

logger = logging.getLogger(__name__)


def is_metadata(fname: str):
    if not os.path.isfile(fname):
        return False, None

    with h5py.File(fname, "r") as f:
        if "/entry/schema_version" in f:
            return True, 'nexus'

    return False, None 


def get_metadata(meta_dir: str):
    meta_fnames = glob.glob(meta_dir + "/*_metadata.hdf")
    if len(meta_fnames) >= 1:
        for f in meta_fnames:
            is_meta, meta_type = is_metadata(f)
            if is_meta:
                return f, meta_type
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


class XpcsResult:
    def __init__(self,
                 meta_dir: str,
                 qmap_fname: str,
                 output_dir: str,
                 overwrite=False,
                 avg_frame=1,
                 stride_frame=1,
                 analysis_type='Multitau') -> None:
        self.meta_dir = meta_dir
        self.qmap_fname = qmap_fname
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.avg_frame = avg_frame
        self.stride_frame = stride_frame
        self.analysis_type = analysis_type
        self.fname = None
        self.G2_fname = None
        self.fname_temp = None
        self.G2_fname_temp = None
        self.success = False

    def __enter__(self):
        """
        Perform setup when entering the context manager.
        """
        meta_fname, meta_ftype = get_metadata(self.meta_dir)
        logger.info(f'metadata filename/type is {meta_fname} | {meta_ftype}')
        self.fname = create_unique_file(self.output_dir, meta_fname,
                                        analysis_type=self.analysis_type,
                                        overwrite=self.overwrite)
        self.G2_fname = os.path.splitext(self.fname)[0] + '_G2.hdf'
        self.fname_temp = self.fname + '.temp'
        self.G2_fname_temp = self.G2_fname + '.temp'

        # Append qmap, metadata, and processing arguments
        result_kwargs = {
            'meta_fname': meta_fname,
            'qmap_fname': self.qmap_fname,
            'meta_type': meta_ftype,
            'avg_frame': self.avg_frame,
            'stride_frame': self.stride_frame,
            'analysis_type': self.analysis_type
        }
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        append_qmap(self.fname_temp, **result_kwargs)
        os.chmod(self.fname_temp, 0o644)
        return self

    def save(self, result_dict, mode="alias", compression=None, **kwargs):
        """
        Save the provided result_dict to the temporary files.
        """
        G2 = result_dict.pop('G2IPIF', None)
        put(self.fname_temp, result_dict, mode=mode,
            compression=compression, **kwargs)

        if G2 is not None:
            put(self.G2_fname_temp, {'G2IPIF': G2}, mode="raw",
                compression=compression, **kwargs)
            # Create a link for the main file
            with h5py.File(self.fname_temp, 'r+') as f:
                relative_path = os.path.basename(self.G2_fname)
                f['/exchange/G2IPIF'] = h5py.ExternalLink(relative_path, 
                                                          "/G2IPIF")

    def __exit__(self, exc_type, exc_value, traceback_obj):
        """
        Exit the context manager, finalizing the temporary files.
        """
        try:
            if os.path.isfile(self.fname_temp):
                shutil.move(self.fname_temp, self.fname)
            if os.path.isfile(self.G2_fname_temp):
                shutil.move(self.G2_fname_temp, self.G2_fname)
            self.success = True
        except Exception:
            traceback.print_exc()
            logger.info('failed to rename the result files')
        finally:
            # Clean up temporary files if they still exist
            for temp_file in [self.fname_temp, self.G2_fname_temp]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.info(f"Cleaned up temporary file: {temp_file}")

        if self.success:
            logger.info(f"Succeeded in saving result file: {self.fname}")
        else:
            logger.error(f"Failed in saving result file: {self.fname}")
            raise
        return self.success


if __name__ == "__main__":
    pass

"""Module for handling XPCS correlation analysis results.
This module defines functions and classes for processing and storing results.
TODO: Add detailed documentation.
"""

import glob
import logging
import os
import shutil
import traceback

import h5py

from boost_corr.xpcs_aps_8idi.append_metadata_qmap import append_metadata_qmap
from boost_corr.xpcs_aps_8idi.hdf_reader import put_results_in_hdf5

logger = logging.getLogger(__name__)


def is_metadata(fname: str):
    """Check if the given file is a metadata file.

    Parameters:
        fname (str): Path to the file.

    Returns:
        tuple[bool, Optional[str]]: A tuple where the first element indicates whether the file is a metadata file,
        and the second element contains metadata information if available, otherwise None.
    """
    if not os.path.isfile(fname):
        return False, None
    # TODO: Implement actual metadata check
    return True, "metadata"


def get_metadata(meta_dir: str):
    """Retrieve the metadata file name from a directory.

    Parameters:
        meta_dir (str): Directory to search for metadata files.

    Returns:
        str: The path to the metadata file.

    Raises:
        FileNotFoundError: If no metadata file is found in the directory.
    """
    meta_fnames = glob.glob(os.path.join(meta_dir, "*_metadata.hdf"))
    if meta_fnames:
        return meta_fnames[0]
    raise FileNotFoundError("Metadata file not found.")


def create_unique_file(
    output_dir: str, meta_fname: str, overwrite: bool = False
) -> str:
    """
    Create a unique filename for an HDF file, avoiding overwriting existing files.

    This function generates a unique filename based on the input parameters. It ensures
    the file has an .hdf extension and avoids overwriting existing files by appending a number if necessary.

    Args:
        output_dir (str): The directory where the file will be created.
        meta_fname (str): The base filename to use.
        overwrite (bool, optional): If True, allows overwriting existing files. Defaults to False.

    Returns:
        str: A unique filename (including path).
    """
    # Create the base filename
    base_fname = os.path.basename(meta_fname)

    # Ensure the filename has the .hdf extension
    name, ext = os.path.splitext(base_fname)
    if name.endswith("_metadata"):
        name = name.removesuffix("_metadata")
    base_fname = name + "_results.hdf"

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
    """Class for storing XPCS correlation analysis results.

    Attributes:
        meta_dir (Optional[str]): Path to the metadata directory.
    """

    def __init__(self, meta_dir: str = None) -> None:
        """Initialize the XpcsResult object.

        Parameters:
            meta_dir (Optional[str]): Directory containing metadata. Defaults to None.

        Returns:
            None
        """
        self.meta_dir = meta_dir
        self.qmap_fname = None
        self.output_dir = None
        self.overwrite = False
        self.fname = None
        self.G2_fname = None
        self.fname_temp = None
        self.G2_fname_temp = None
        self.success = True
        self.analysis_config = {
            "multitau_config": {},
            "twotime_config": {},
        }

    def __enter__(self):
        """
        Perform setup when entering the context manager.
        """
        meta_fname, meta_ftype = get_metadata(self.meta_dir)
        logger.info(f"metadata filename/type is {meta_fname} | {meta_ftype}")
        self.fname = create_unique_file(
            self.output_dir, meta_fname, overwrite=self.overwrite
        )
        self.G2_fname = os.path.splitext(self.fname)[0] + "_G2.hdf"
        self.fname_temp = self.fname + ".temp"
        self.G2_fname_temp = self.G2_fname + ".temp"

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        append_metadata_qmap(self.fname_temp, meta_fname, self.qmap_fname)
        os.chmod(self.fname_temp, 0o644)
        self.append(self.analysis_config)
        return self

    def append(self, result_dict):
        """
        Save the provided result_dict to the temporary files.
        """
        try:
            if "G2" in result_dict:
                self.append_and_link_G2(result_dict["G2"])
                del result_dict["G2"]
            put_results_in_hdf5(self.fname_temp, result_dict)
        except Exception:
            traceback.print_exc()
            self.success = False
            raise
        else:
            self.success = True and self.success

    def append_and_link_G2(self, G2_dict):
        """
        append the provided result_dict to the temporary files.
        """
        try:
            put_results_in_hdf5(self.G2_fname_temp, {"G2IPIF": G2_dict}, mode="raw")
            # Create a link for the main file
            with h5py.File(self.fname_temp, "r+") as f:
                relative_path = os.path.basename(self.G2_fname)
                f["/xpcs/multitau/unnormalized_g2"] = h5py.ExternalLink(
                    relative_path, "/G2IPIF"
                )
        except Exception:
            traceback.print_exc()
            self.success = False
            raise
        else:
            self.success = True and self.success

    def __exit__(self, exc_type, exc_value, traceback_obj):
        """
        Exit the context manager, finalizing the temporary files.
        """
        if self.success:
            try:
                if os.path.isfile(self.fname_temp):
                    shutil.move(self.fname_temp, self.fname)
                if os.path.isfile(self.G2_fname_temp):
                    shutil.move(self.G2_fname_temp, self.G2_fname)
            except Exception:
                traceback.print_exc()
                logger.error("failed to rename the result files")
                self.success = False
                raise

        # try to clean up regardless of success
        for temp_file in [self.fname_temp, self.G2_fname_temp]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info(f"Cleaned up temporary file: {temp_file}")

        if self.success:
            logger.info(f"Succeeded in saving result file: {self.fname}")
        else:
            logger.error(f"Failed in saving result file: {self.fname}")
        return self.success


if __name__ == "__main__":
    pass

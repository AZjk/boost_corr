import glob
import logging
import os
import shutil
import traceback
import h5py

from .append_metadata_qmap import append_metadata_qmap
from .hdf_reader import put_results_in_hdf5

logger = logging.getLogger(__name__)


def is_metadata(fname: str):
    if not os.path.isfile(fname):
        return False, None

    with h5py.File(fname, "r") as f:
        if "/entry/schema_version" in f:
            return True, "nexus"

    return False, None


def get_metadata(meta_dir: str):
    meta_fnames = glob.glob(meta_dir + "/*_metadata.hdf")
    if len(meta_fnames) >= 1:
        for f in meta_fnames:
            is_meta, meta_type = is_metadata(f)
            if is_meta:
                return f, meta_type
    raise FileNotFoundError(f"no metadata file found in [{meta_dir}]")


def create_unique_file(
    output_dir: str,
    meta_fname: str,
    overwrite: bool = False,
    prefix: str = None,
    suffix: str = None,
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
    name = name.rstrip("_metadata")

    if prefix:
        name = f"{prefix.rstrip('_')}_{name}"
    if suffix:
        name = f"{name}_{suffix.lstrip('_')}"

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
    def __init__(
        self,
        meta_dir=None,
        qmap_fname=None,
        output_dir=None,
        overwrite=False,
        rawdata_path=None,
        multitau_config=None,
        twotime_config=None,
        prefix=None,
        suffix=None,
    ) -> None:
        self.meta_dir = meta_dir
        self.qmap_fname = qmap_fname
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.fname = None
        self.G2_fname = None
        self.fname_temp = None
        self.G2_fname_temp = None
        self.success = True
        self.prefix = prefix
        self.suffix = suffix
        self.analysis_config = {
            "rawdata_path": rawdata_path,
            "multitau_config": multitau_config or {},
            "twotime_config": twotime_config or {},
        }

    def __enter__(self):
        """
        Perform setup when entering the context manager.
        """
        meta_fname, meta_ftype = get_metadata(self.meta_dir)
        logger.info(f"metadata filename/type is {meta_fname} | {meta_ftype}")
        self.fname = create_unique_file(
            self.output_dir,
            meta_fname,
            overwrite=self.overwrite,
            prefix=self.prefix,
            suffix=self.suffix,
        )
        self.G2_fname = os.path.splitext(self.fname)[0] + "_G2.hdf"
        self.fname_temp = self.fname + ".temp"
        self.G2_fname_temp = self.G2_fname + ".temp"

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        append_metadata_qmap(self.fname_temp, meta_fname, self.qmap_fname)
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

    def correct_t0_for_timepix4(self, t0):
        """
        Correct the t0 value in the multitau configuration for Timepix4 data.
        """
        logger.info(f"Correcting t0 for Timepix4 data. {t0=} s")
        with h5py.File(self.fname_temp, "r+") as f:
            if "/entry/instrument/detector_1/frame_time" in f:
                del f["/entry/instrument/detector_1/frame_time"]
            f["/entry/instrument/detector_1/frame_time"] = t0

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

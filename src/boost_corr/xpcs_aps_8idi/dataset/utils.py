import os
import h5py
from .imm_handler import ImmDataset
from .rigaku_handler import RigakuDataset
from .rigaku_3M_handler import Rigaku3MDataset
from .hdf_handler import HdfDataset


def create_dataset(
    raw_fname,
    device,
    mask_crop,
    avg_frame,
    begin_frame,
    end_frame,
    stride_frame,
    bin_time_s=1e-6,
    run_config_path=None,
):
    if not os.path.isfile(raw_fname):
        raise FileNotFoundError(f"The raw_file '{raw_fname}' does not exist.")

    ext = os.path.splitext(raw_fname)[-1]
    # use_loader is set for HDF files, it can use multiple processes to read
    # large HDF file;
    use_loader = False
    if ext == ".bin":
        dataset_method = RigakuDataset
        use_loader = False
        batch_size = 1024
    elif raw_fname.endswith(".bin.000"):
        dataset_method = Rigaku3MDataset
        use_loader = False
        batch_size = 256
    elif ext in [".imm", ".h5", ".hdf", ".hdf5"]:
        # Check if it's an HDF5 file by trying to detect HDF5 format
        # If not HDF5, treat it as IMM format
        real_raw = os.path.realpath(raw_fname)
        try:
            is_hdf5 = h5py.is_hdf5(real_raw)
        except (OSError, IOError):
            # File might be corrupted or inaccessible
            is_hdf5 = False
        
        if is_hdf5:
            dataset_method = HdfDataset
            use_loader = True
            batch_size = 8
        else:
            dataset_method = ImmDataset
            use_loader = False
            batch_size = 8
    elif raw_fname.endswith((".tpx", ".tpx.000", ".tpx.001", ".tpx.002")):
        from .timepix4_handler import TimepixAnalysisDataset

        dataset_method = TimepixAnalysisDataset
        use_loader = False
        batch_size = 1024
    else:
        raise TypeError(f"File type [{ext}] is not supported")

    dset = dataset_method(
        raw_fname,
        batch_size=batch_size,
        device=device,
        mask_crop=mask_crop,
        avg_frame=avg_frame,
        begin_frame=begin_frame,
        end_frame=end_frame,
        stride_frame=stride_frame,
        bin_time_s=bin_time_s,
        run_config_path=run_config_path,
    )

    return dset, use_loader

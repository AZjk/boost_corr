"""Module for dataset utilities.

This module provides functionality to create datasets from raw data files.
"""

import os

import magic

from boost_corr.xpcs_aps_8idi.dataset.hdf_handler import HdfDataset
from boost_corr.xpcs_aps_8idi.dataset.imm_handler import ImmDataset
from boost_corr.xpcs_aps_8idi.dataset.rigaku_3M_handler import Rigaku3MDataset
from boost_corr.xpcs_aps_8idi.dataset.rigaku_handler import RigakuDataset


def create_dataset(
    raw_fname, device, mask_crop, avg_frame, begin_frame, end_frame, stride_frame
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
    elif ext in [".imm", ".h5", ".hdf"]:
        real_raw = os.path.realpath(raw_fname)
        ftype = magic.from_file(real_raw)
        if ftype == "empty":
            raise Exception("The raw file is damaged.")
        elif ftype == "Hierarchical Data Format (version 5) data":
            dataset_method = HdfDataset
            use_loader = True
            batch_size = 8
        else:
            dataset_method = ImmDataset
            use_loader = False
            batch_size = 8
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
    )

    return dset, use_loader

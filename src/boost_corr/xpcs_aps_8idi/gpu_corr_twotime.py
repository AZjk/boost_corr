import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

import boost_corr.xpcs_aps_8idi.exceptions as exc

from .. import TwotimeCorrelator
from ..help_functions import get_device
from .dataset import create_dataset
from .xpcs_qpartitionmap import XpcsQPartitionMap
from .xpcs_result import XpcsResult

logger = logging.getLogger(__name__)


def solve_twotime(*args: Any, **kwargs: Any) -> Union[str, None]:
    kwargs_record = kwargs.copy()
    kwargs_record["analysis_type"] = "twotime"
    num_rawfiles = len(kwargs["raw"])
    if num_rawfiles == 1:
        kwargs["raw"] = kwargs["raw"][0]
        return solve_twotime_base(*args, analysis_kwargs=kwargs_record, **kwargs)
    else:
        all_rawfiles = kwargs["raw"].copy()
        for raw in all_rawfiles:
            kwargs["raw"] = raw
            solve_twotime_base(*args, analysis_kwargs=kwargs_record, **kwargs)
        return


def solve_twotime_base(
    qmap: Union[str, Path] = None,
    raw: Union[str, Path] = None,
    output: str = "cluster_results",
    batch_size: int = 8,
    gpu_id: int = 0,
    verbose: bool = False,
    crop_ratio_threshold: float = 0.5,
    num_loaders: int = 16,
    begin_frame: int = 0,
    end_frame: int = -1,
    avg_frame: int = 1,
    stride_frame: int = 1,
    overwrite: bool = False,
    # save_G2: bool = False,
    dq_selection: Optional[Union[str, Path]] = None,
    smooth: str = "sqmap",
    analysis_kwargs: Optional[dict] = None,
    save_results: bool = True,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    meta_fname: Optional[str] = None,
    **kwargs,
):

    log_level = logging.INFO if verbose else logging.ERROR
    logger.setLevel(log_level)
    device = get_device(gpu_id)

    # force crop to use only valid pixels; needed for twotime correlator
    logger.info(f"forcing cropping to use only valid pixels: {crop_ratio_threshold=:.2f} -> 1.0")
    crop_ratio_threshold = 1.0

    # create qpartitionmap
    try:
        qpm = XpcsQPartitionMap(
            qmap,
            device=device,
            flag_sort=True,
            crop_ratio_threshold=crop_ratio_threshold,
            dq_selection=dq_selection,
        )
    except Exception as e:
        raise exc.QMapError from e
    if verbose:
        qpm.describe()
        logger.info(f"device: {device}")

    # create dataset
    try:
        dset, use_loader = create_dataset(
            raw,
            device=device,
            mask_crop=qpm.mask_crop,
            avg_frame=avg_frame,
            begin_frame=begin_frame,
            end_frame=end_frame,
            stride_frame=stride_frame,
        )
    except Exception as e:
        raise exc.DatasetError from e

    # in some detectors/configurations, the qmap is rotated
    qpm.update_rotation(dset.det_size)

    try:
        twotime_correlator = TwotimeCorrelator(
            qpm.qinfo,
            frame_num=dset.frame_num,
            det_size=dset.det_size,
            method="normal",
            mask_crop=qpm.mask_crop,
            window=1024,
            device=device,
        )
    except Exception as e:
        raise exc.CorrelatorError from e

    logger.info("correlation solver created.")

    t_start = time.perf_counter()
    try:
        twotime_correlator.process_dataset(dset, verbose=verbose, use_loader=use_loader, num_workers=num_loaders)
        twotime_correlator.post_processing(smooth_method=smooth)
    except Exception as e:
        raise exc.ProcessingError from e
    t_end = time.perf_counter()
    t_diff = t_end - t_start
    frequency = dset.frame_num / t_diff
    logger.info(f"correlation finished in {t_diff:.2f}s." + f" frequency = {frequency:.2f} Hz")

    t_start = time.perf_counter()
    try:
        raw_scattering = twotime_correlator.get_scattering()
        norm_scattering = qpm.normalize_scattering(raw_scattering)
    except Exception as e:
        raise exc.PostProcessingError from e
    t_end = time.perf_counter()
    logger.info("normalization finished in %.3fs" % (t_end - t_start))

    # saving results to file
    if save_results:
        try:
            with XpcsResult(
                raw_fname=raw,
                qmap_fname=qmap,
                output_dir=output,
                meta_fname=meta_fname,
                overwrite=overwrite,
                twotime_config=analysis_kwargs,
                prefix=prefix,
                suffix=suffix,
            ) as result_file:
                result_file.append(norm_scattering)
                for c2_payload in twotime_correlator.get_twotime_generator():
                    result_file.append(c2_payload)

            logger.info(f"twotime analysis finished")
            return result_file.fname
        except Exception as e:
            raise exc.ResultSavingError from e
    else:
        result_file_kwargs = {
            "raw_fname": raw,
            "meta_fname": meta_fname,
            "qmap_fname": qmap,
            "output_dir": output,
            "overwrite": overwrite,
            "twotime_config": analysis_kwargs,
            "prefix": prefix,
            "suffix": suffix,
        }
        return result_file_kwargs, (
            norm_scattering,
            twotime_correlator.get_twotime_generator(),
        )

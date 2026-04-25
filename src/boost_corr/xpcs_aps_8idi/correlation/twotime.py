import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import boost_corr.xpcs_aps_8idi.exceptions as exc

from ...correlator.twotime import TwotimeCorrelator
from ...help_functions import get_device
from ..dataset import create_dataset
from ..xpcs_result import XpcsResult, check_metadata
from .common import (
    attach_debug_note,
    create_qpm,
    empty_gpu_cache,
    load_raw_list,
)

logger = logging.getLogger(__name__)


def _create_correlator(qpm, dset, device):
    try:
        return TwotimeCorrelator(
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


def _run_correlation(correlator, dset, smooth, verbose, use_loader, num_loaders):
    t_start = time.perf_counter()
    try:
        correlator.process_dataset(
            dset, verbose=verbose, use_loader=use_loader, num_workers=num_loaders
        )
        correlator.post_processing(smooth_method=smooth)
    except Exception as e:
        raise exc.ProcessingError from e
    t_diff = time.perf_counter() - t_start
    logger.info(
        f"correlation finished in {t_diff:.2f}s. frequency = {dset.frame_num / t_diff:.2f} Hz"
    )


def _normalize_results(correlator, qpm):
    t_start = time.perf_counter()
    try:
        raw_scattering = correlator.get_scattering()
        norm_scattering = qpm.normalize_scattering(raw_scattering)
        twotime_gen = correlator.get_twotime_generator()
    except Exception as e:
        raise exc.PostProcessingError from e
    logger.info("normalization finished in %.3fs" % (time.perf_counter() - t_start))
    return norm_scattering, twotime_gen


def _save_result(result_kwargs, norm_scattering, twotime_gen):
    try:
        with XpcsResult(**result_kwargs) as result_file:
            result_file.append(norm_scattering)
            for c2_payload in twotime_gen:
                result_file.append(c2_payload)
        logger.info("twotime analysis finished")
        return result_file.fname
    except Exception as e:
        raise exc.ResultSavingError from e


def solve_twotime(
    qmap: Union[str, Path] = None,
    raw: list = None,
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
    dq_selection: Optional[Union[str, Path]] = None,
    smooth: str = "sqmap",
    save_results: bool = True,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    meta_fname: Optional[str] = None,
    **kwargs: Any,
) -> Union[str, None]:
    log_level = logging.INFO if verbose else logging.ERROR
    logger.setLevel(log_level)

    # force crop to use only valid pixels; needed for twotime correlator
    logger.info(
        f"forcing cropping to use only valid pixels: {crop_ratio_threshold=:.2f} -> 1.0"
    )
    crop_ratio_threshold = 1.0

    analysis_kwargs = {
        k: v for k, v in locals().items() if k not in ("save_results", "kwargs")
    }
    analysis_kwargs["analysis_type"] = "twotime"

    device = get_device(gpu_id)
    qpm = create_qpm(
        qmap,
        device,
        crop_ratio_threshold,
        dq_selection=dq_selection,
        flag_sort=True,
    )

    if verbose:
        qpm.describe()
        logger.info(f"device: {device}")

    common_dset_kwargs = {
        "device": device,
        "mask_crop": qpm.mask_crop,
        "avg_frame": avg_frame,
        "begin_frame": begin_frame,
        "end_frame": end_frame,
        "stride_frame": stride_frame,
    }

    raw = load_raw_list(raw)

    jobs = [(raw_fname, common_dset_kwargs.copy(), suffix) for raw_fname in raw]
    if len(jobs) > 1:
        logger.info(f"twotime batch process, num_rawfiles: {len(jobs)}")

    single_job = len(jobs) == 1
    n_jobs = len(jobs)
    correlator = None
    last_fname = None
    failed_jobs = []

    for job_idx, (raw_fname, dset_kwargs, _suffix) in enumerate(jobs, start=1):
        t_job_start = time.perf_counter()
        try:
            check_metadata(raw_fname, meta_fname)
            dset, use_loader = create_dataset(raw_fname, **dset_kwargs)
            # in some detectors/configurations, the qmap is rotated;
            # twotime needs this resolved before correlator construction
            qpm.update_rotation(dset.det_size)

            if correlator is not None:
                logger.info("freeing existing correlator to reclaim VRAM")
                correlator = None
                empty_gpu_cache(device)
            correlator = _create_correlator(qpm, dset, device)

            if verbose:
                logger.info("correlation solver created.")

            _run_correlation(
                correlator, dset, smooth, verbose, use_loader, num_loaders
            )
            norm_scattering, twotime_gen = _normalize_results(correlator, qpm)

            result_kwargs = {
                "raw_fname": raw_fname,
                "qmap_fname": qmap,
                "output_dir": output,
                "meta_fname": meta_fname,
                "overwrite": overwrite,
                "twotime_config": analysis_kwargs,
                "prefix": prefix,
                "suffix": _suffix,
            }

            if not save_results and single_job:
                return result_kwargs, (norm_scattering, twotime_gen)

            last_fname = _save_result(result_kwargs, norm_scattering, twotime_gen)
            elapsed = time.perf_counter() - t_job_start
            ts = datetime.now().strftime("%m-%d %H:%M:%S")
            print(f"[{ts}] [{job_idx}/{n_jobs}] ({elapsed:.1f}s) saved: {last_fname}")
        except Exception as e:
            attach_debug_note(e, analysis_kwargs, raw_fname)
            if single_job:
                raise
            elapsed = time.perf_counter() - t_job_start
            ts = datetime.now().strftime("%m-%d %H:%M:%S")
            print(f"[{ts}] [{job_idx}/{n_jobs}] ({elapsed:.1f}s) FAILED: {raw_fname}")
            logger.error(f"job failed for {raw_fname}: {e}", exc_info=True)
            failed_jobs.append(raw_fname)
            correlator = None  # reset: state may be corrupted after a failed job

    if failed_jobs:
        logger.warning(f"{len(failed_jobs)} job(s) failed: {failed_jobs}")

    return last_fname

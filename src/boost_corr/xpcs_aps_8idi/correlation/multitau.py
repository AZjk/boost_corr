import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import boost_corr.xpcs_aps_8idi.exceptions as exc

from ...correlator.multitau import MultitauCorrelator
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


def _build_segment_jobs(
    raw_file, num_segments, begin_frame, suffix, common_dset_kwargs
):
    """Return list of (raw_fname, dset_kwargs, suffix) for all segments of one file."""
    assert common_dset_kwargs["stride_frame"] == 1, (
        "multiple segments process only supports stride frame = 1"
    )
    assert common_dset_kwargs["avg_frame"] == 1, (
        "multiple segments process only supports avg frame = 1"
    )
    dset, _ = create_dataset(raw_file, **common_dset_kwargs)
    total_frame_num = dset.frame_num
    assert total_frame_num % num_segments == 0, (
        "total frame number must be divisible by num_segments"
    )
    segment_frame_num = total_frame_num // num_segments
    assert segment_frame_num >= 1, "segment frame number must be >= 1"

    jobs = []
    suffix_width = len(str(num_segments - 1))
    for n in range(num_segments):
        entry = common_dset_kwargs.copy()
        entry["begin_frame"] = begin_frame + n * segment_frame_num
        entry["end_frame"] = begin_frame + (n + 1) * segment_frame_num
        _suffix = f"segment_{n:0{suffix_width}d}"
        if suffix is not None:
            _suffix = f"{suffix}_{_suffix}"
        jobs.append((raw_file, entry, _suffix))
    return jobs


def _create_or_reset_correlator(existing, dset, mask_crop, multitau_kwargs):
    if (
        existing is None
        or existing.frame_num != dset.frame_num
        or existing.det_size != dset.det_size
    ):
        try:
            return MultitauCorrelator(
                dset.det_size,
                frame_num=dset.frame_num,
                mask_crop=mask_crop,
                **multitau_kwargs,
            )
        except Exception as e:
            raise exc.CorrelatorError from e
    else:
        logger.info("reset correlator")
        existing.reset()
        return existing


def _run_correlation(correlator, dset, verbose, use_loader, num_loaders):
    t_start = time.perf_counter()
    try:
        correlator.process_dataset(
            dset, verbose=verbose, use_loader=use_loader, num_workers=num_loaders
        )
    except Exception as e:
        raise exc.ProcessingError from e
    t_diff = time.perf_counter() - t_start
    logger.info(
        f"correlation finished in {t_diff:.2f}s. frequency = {dset.frame_num / t_diff:.2f} Hz"
    )


def _normalize_results(correlator, qpm, save_G2):
    t_start = time.perf_counter()
    try:
        output_scattering, output_multitau = correlator.get_results()
        norm_scattering = qpm.normalize_scattering(output_scattering)
        norm_multitau = qpm.normalize_multitau(output_multitau, save_G2=save_G2)
        part_multitau = correlator.get_partial_g2()
    except Exception as e:
        raise exc.PostProcessingError from e
    logger.info("normalization finished in %.3fs" % (time.perf_counter() - t_start))
    return norm_scattering, norm_multitau, part_multitau


def _save_result(
    result_kwargs,
    norm_scattering,
    norm_multitau,
    part_multitau,
    dset,
    bin_time_s,
):
    try:
        with XpcsResult(**result_kwargs) as result_file:
            result_file.append(norm_scattering)
            result_file.append(norm_multitau)
            result_file.append(part_multitau)
            if dset.dataset_type == "Timepix4Dataset":
                result_file.correct_t0_for_timepix4(bin_time_s)
        logger.info("multitau analysis finished")
        return result_file.fname
    except Exception as e:
        raise exc.ResultSavingError from e


def solve_multitau(
    qmap: Union[str, Path] = None,
    raw: list = None,
    output: str = "cluster_results",
    batch_size: int = 8,
    gpu_id: int = 0,
    verbose: bool = False,
    crop_ratio_threshold: float = 0.5,
    num_loaders: int = 16,
    normalize_frame: bool = False,
    begin_frame: int = 0,
    end_frame: int = -1,
    avg_frame: int = 1,
    stride_frame: int = 1,
    overwrite: bool = False,
    save_G2: bool = False,
    save_results: bool = True,
    num_partial_g2: int = 0,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    bin_time_s: float = 1e-6,
    run_config_path=None,
    max_memory: float = 36.0,
    num_segments: int = 1,
    meta_fname: Optional[str] = None,
    **kwargs: Any,
) -> Union[str, None]:
    analysis_kwargs = {
        k: v for k, v in locals().items() if k not in ("save_results", "kwargs")
    }
    analysis_kwargs["analysis_type"] = "multitau"

    device = get_device(gpu_id)
    qpm = create_qpm(qmap, device, crop_ratio_threshold)

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
        "bin_time_s": bin_time_s,
        "run_config_path": run_config_path,
    }

    multitau_kwargs = {
        "queue_size": batch_size,
        "auto_queue": True,
        "device": device,
        "normalize_frame": normalize_frame,
        "qpm": qpm,
        "num_partial_g2": num_partial_g2,
        "max_memory": max_memory,
    }

    raw = load_raw_list(raw)

    if num_segments > 1:
        logger.info(
            f"multiple segments process, num_segments: {num_segments}, num_rawfiles: {len(raw)}"
        )
        jobs = []
        for raw_file in raw:
            jobs.extend(
                _build_segment_jobs(
                    raw_file, num_segments, begin_frame, suffix, common_dset_kwargs
                )
            )
    elif len(raw) == 1:
        jobs = [(raw[0], common_dset_kwargs, suffix)]
    else:
        logger.info(
            f"single segment process for multiple files, num_rawfiles: {len(raw)}"
        )
        jobs = [(raw_fname, common_dset_kwargs.copy(), suffix) for raw_fname in raw]

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
            qpm.update_rotation(dset.det_size)
            if correlator is not None and (
                correlator.frame_num != dset.frame_num
                or correlator.det_size != dset.det_size
            ):
                logger.info("freeing existing correlator to reclaim VRAM")
                correlator = None
                empty_gpu_cache(device)
            correlator = _create_or_reset_correlator(
                correlator,
                dset,
                qpm.mask_crop,
                multitau_kwargs,
            )

            if verbose:
                dset.describe()
                correlator.describe()
                logger.info("correlation solver created.")

            _run_correlation(correlator, dset, verbose, use_loader, num_loaders)
            norm_scattering, norm_multitau, part_multitau = _normalize_results(
                correlator, qpm, save_G2
            )

            result_kwargs = {
                "raw_fname": raw_fname,
                "qmap_fname": qmap,
                "output_dir": output,
                "meta_fname": meta_fname,
                "overwrite": overwrite,
                "multitau_config": analysis_kwargs,
                "prefix": prefix,
                "suffix": _suffix,
            }

            if not save_results and single_job:
                return result_kwargs, (norm_scattering, norm_multitau)

            last_fname = _save_result(
                result_kwargs,
                norm_scattering,
                norm_multitau,
                part_multitau,
                dset,
                bin_time_s,
            )
            elapsed = time.perf_counter() - t_job_start
            ts = datetime.now().strftime("%m-%d %H:%M:%S")
            print(f"[{ts}] [{job_idx}/{n_jobs}] ({elapsed:.1f}s) saved: {last_fname}")
        except Exception as e:
            attach_debug_note(
                e,
                analysis_kwargs,
                raw_fname,
                extra_keys=("normalize_frame", "num_segments"),
            )
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

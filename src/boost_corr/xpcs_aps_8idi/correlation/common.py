"""Correlation solver and helpers for APS 8-ID-I XPCS data."""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import torch

import boost_corr.xpcs_aps_8idi.exceptions as exc

from ...correlator.multitau import MultitauCorrelator
from ...correlator.twotime import TwotimeCorrelator
from ...devices import get_device
from ..dataset import create_dataset
from ..xpcs_qpartitionmap import XpcsQPartitionMap
from ..xpcs_result import XpcsResult, check_metadata

logger = logging.getLogger(__name__)


# Keys surfaced in exception notes when a single job fails inside a batch.
# Callers add analysis-specific keys via attach_debug_note(extra_keys=...).
_BASE_DEBUG_KEYS = frozenset(
    {
        "raw",
        "qmap",
        "output",
        "meta_fname",
        "gpu_id",
        "begin_frame",
        "end_frame",
        "avg_frame",
        "stride_frame",
    }
)


def create_qpm(qmap, device, crop_ratio_threshold, dq_selection=None, flag_sort=False):
    try:
        return XpcsQPartitionMap(
            qmap,
            device=device,
            flag_sort=flag_sort,
            crop_ratio_threshold=crop_ratio_threshold,
            dq_selection=dq_selection,
        )
    except Exception as e:
        raise exc.QMapError from e


def empty_gpu_cache(device: str):
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device.startswith("xpu"):
        torch.xpu.empty_cache()


def load_raw_list(raw):
    """Return raw as a flat list of file paths.

    If the input is a single .txt file, read paths from it (one per non-empty line).
    """
    if len(raw) == 1 and raw[0].endswith(".txt"):
        logger.info(f"raw input is a text file, loading raw file list from {raw[0]}")
        with open(raw[0], "r") as f:
            raw = [line.strip() for line in f if line.strip()]
    return raw


def attach_debug_note(e, analysis_kwargs, raw_fname, extra_keys=()):
    """Attach an analysis_kwargs note to e for batch-failure diagnostics."""
    keys = _BASE_DEBUG_KEYS | set(extra_keys)
    debug_info = {k: analysis_kwargs[k] for k in keys if k in analysis_kwargs}
    debug_info["raw"] = raw_fname
    e.add_note(f"analysis_kwargs: {debug_info}")


def create_or_reset_correlator(existing, dset, factory, device):
    """Reuse correlator via reset() if shape matches dset; else free and rebuild via factory."""
    if existing is not None and (
        existing.frame_num != dset.frame_num
        or existing.det_size != dset.det_size
    ):
        logger.info("freeing existing correlator to reclaim VRAM")
        existing = None
        empty_gpu_cache(device)

    if existing is None:
        try:
            return factory()
        except Exception as e:
            raise exc.CorrelatorError from e

    logger.info("reset correlator")
    existing.reset()
    return existing


def run_correlation(
    correlator, dset, verbose, use_loader, num_loaders, post_kwargs=None
):
    """Run process_dataset + post_process with timing/error wrapping."""
    if post_kwargs is None:
        post_kwargs = {}
    t_start = time.perf_counter()
    try:
        correlator.process_dataset(
            dset, verbose=verbose, use_loader=use_loader, num_workers=num_loaders
        )
        correlator.post_process(**post_kwargs)
    except Exception as e:
        raise exc.ProcessingError from e
    t_diff = time.perf_counter() - t_start
    logger.info(
        f"correlation finished in {t_diff:.2f}s. frequency = {dset.frame_num / t_diff:.2f} Hz"
    )


def normalize_results(correlator, qpm, **opts):
    """Return an iterable of normalized payload dicts, dispatched on the correlator."""
    t_start = time.perf_counter()
    try:
        payloads = correlator.get_normalized_payloads(qpm, **opts)
    except Exception as e:
        raise exc.PostProcessingError from e
    logger.info("normalization finished in %.3fs" % (time.perf_counter() - t_start))
    return payloads


def save_result(result_kwargs, payloads, label, post_save=None):
    """Write all payloads to a result file. post_save runs after appending, before close."""
    try:
        with XpcsResult(**result_kwargs) as result_file:
            for payload in payloads:
                result_file.append(payload)
            if post_save is not None:
                post_save(result_file)
        logger.info(f"{label} analysis finished")
        return result_file.fname
    except Exception as e:
        raise exc.ResultSavingError from e


def run_jobs(
    jobs,
    *,
    qpm,
    device,
    label,
    config_key,
    analysis_kwargs,
    result_kwargs_extras,
    correlator_factory,
    save_results,
    meta_fname,
    num_loaders,
    verbose,
    correlation_post_kwargs=None,
    normalize_opts=None,
    save_post_save=None,
    debug_extra_keys=(),
):
    """Run the per-job correlation loop used by solve_correlation.

    correlator_factory: callable(dset) -> correlator
    save_post_save: callable(result_file, dset) -> None, or None
    """
    if normalize_opts is None:
        normalize_opts = {}
    single_job = len(jobs) == 1
    n_jobs = len(jobs)
    correlator = None
    failed_jobs = []

    for job_idx, (raw_fname, dset_kwargs, _suffix) in enumerate(jobs, start=1):
        t_job_start = time.perf_counter()
        try:
            check_metadata(raw_fname, meta_fname)
            dset, use_loader = create_dataset(raw_fname, **dset_kwargs)
            qpm.update_rotation(dset.det_size)
            correlator = create_or_reset_correlator(
                correlator, dset, lambda: correlator_factory(dset), device
            )

            if verbose:
                if hasattr(dset, "describe"):
                    dset.describe()
                if hasattr(correlator, "describe"):
                    correlator.describe()
                logger.info("correlation solver created.")

            run_correlation(
                correlator,
                dset,
                verbose,
                use_loader,
                num_loaders,
                post_kwargs=correlation_post_kwargs,
            )
            payloads = normalize_results(correlator, qpm, **normalize_opts)

            result_kwargs = {
                "raw_fname": raw_fname,
                "meta_fname": meta_fname,
                config_key: analysis_kwargs,
                "suffix": _suffix,
                **result_kwargs_extras,
            }

            if not save_results and single_job:
                return result_kwargs, payloads

            post_save = (
                (lambda rf: save_post_save(rf, dset))
                if save_post_save is not None
                else None
            )
            fname = save_result(
                result_kwargs, payloads, label=label, post_save=post_save
            )
            elapsed = time.perf_counter() - t_job_start
            log_job_status(job_idx, n_jobs, elapsed, f"saved: {fname}")
        except Exception as e:
            attach_debug_note(
                e, analysis_kwargs, raw_fname, extra_keys=debug_extra_keys
            )
            if single_job:
                raise
            elapsed = time.perf_counter() - t_job_start
            log_job_status(job_idx, n_jobs, elapsed, f"FAILED: {raw_fname}")
            logger.error(f"job failed for {raw_fname}: {e}", exc_info=True)
            failed_jobs.append(raw_fname)
            correlator = None  # reset: state may be corrupted after a failed job

    if failed_jobs:
        logger.warning(f"{len(failed_jobs)} job(s) failed: {failed_jobs}")


def log_job_status(job_idx, n_jobs, elapsed, status):
    """Print a timestamped per-job status line for batch runs."""
    ts = datetime.now().strftime("%m-%d %H:%M:%S")
    print(f"[{ts}] [{job_idx}/{n_jobs}] ({elapsed:.1f}s) {status}")


def build_segment_jobs(raw_file, num_segments, begin_frame, suffix, common_dset_kwargs):
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


def build_jobs(raw, num_segments, begin_frame, suffix, common_dset_kwargs, label=""):
    """Build the full job list. Splits each raw file into segments if num_segments > 1."""
    if num_segments > 1:
        logger.info(
            f"{label}multi-segment process, num_segments: {num_segments}, num_rawfiles: {len(raw)}"
        )
        jobs = []
        for raw_file in raw:
            jobs.extend(
                build_segment_jobs(
                    raw_file, num_segments, begin_frame, suffix, common_dset_kwargs
                )
            )
        return jobs
    if len(raw) == 1:
        return [(raw[0], common_dset_kwargs, suffix)]
    logger.info(
        f"{label}single segment process for multiple files, num_rawfiles: {len(raw)}"
    )
    return [(raw_fname, common_dset_kwargs.copy(), suffix) for raw_fname in raw]


_CORRELATOR_CLASSES = {
    "Multitau": MultitauCorrelator,
    "Twotime": TwotimeCorrelator,
}

_VALID_TYPES = ("Multitau", "Twotime", "Both")


def _build_run_args(analysis_type, *, qpm, device, verbose, num_loaders,
                    meta_fname, save_results, qmap, output, overwrite, prefix,
                    batch_size, normalize_frame, num_partial_g2, max_memory,
                    save_G2, bin_time_s, smooth, analysis_kwargs,
                    skip_scattering=False):
    """Build the correlator factory and run_jobs kwargs for a single analysis type."""
    label = analysis_type.lower()
    CorrelatorClass = _CORRELATOR_CLASSES[analysis_type]
    factory_kwargs = {
        "batch_size": batch_size,
        "normalize_frame": normalize_frame,
        "num_partial_g2": num_partial_g2,
        "max_memory": max_memory,
    }

    def _factory(dset):
        return CorrelatorClass.from_config(qpm, dset, device, **factory_kwargs)

    run_opts: dict = {}
    if analysis_type == "Multitau":
        run_opts["normalize_opts"] = {"save_G2": save_G2,
                                      "skip_scattering": skip_scattering}

        def _post_save(rf, dset):
            if dset.dataset_type == "Timepix4Dataset":
                rf.correct_t0_for_timepix4(bin_time_s)

        run_opts["save_post_save"] = _post_save
        run_opts["debug_extra_keys"] = ("normalize_frame", "num_segments")
    elif analysis_type == "Twotime":
        run_opts["correlation_post_kwargs"] = {"smooth_method": smooth}
        run_opts["normalize_opts"] = {"skip_scattering": skip_scattering}
        run_opts["debug_extra_keys"] = ("num_segments",)

    return dict(
        qpm=qpm,
        device=device,
        label=label,
        config_key=f"{label}_config",
        analysis_kwargs=analysis_kwargs,
        result_kwargs_extras={
            "qmap_fname": qmap,
            "output_dir": output,
            "overwrite": overwrite,
            "prefix": prefix,
        },
        correlator_factory=_factory,
        save_results=save_results,
        meta_fname=meta_fname,
        num_loaders=num_loaders,
        verbose=verbose,
        **run_opts,
    )


def solve_correlation(
    analysis_type: str = "Multitau",
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
    save_results: bool = True,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    num_segments: int = 1,
    meta_fname: Optional[str] = None,
    # multitau-specific
    normalize_frame: bool = False,
    save_G2: bool = False,
    num_partial_g2: int = 0,
    bin_time_s: float = 1e-6,
    run_config_path=None,
    max_memory: float = 36.0,
    # twotime-specific
    dq_selection: Optional[Union[str, Path]] = None,
    smooth: str = "sqmap",
    **kwargs: Any,
) -> None:
    if analysis_type not in _VALID_TYPES:
        raise ValueError(
            f"Unknown analysis_type {analysis_type!r}. Expected one of {_VALID_TYPES}"
        )

    log_level = logging.INFO if verbose else logging.ERROR
    logger.setLevel(log_level)

    analysis_kwargs = {
        k: v
        for k, v in locals().items()
        if k not in ("save_results", "kwargs")
    }

    device = get_device(gpu_id)

    # twotime needs flag_sort and forces crop_ratio_threshold=1.0
    need_twotime = analysis_type in ("Twotime", "Both")
    qpm_kwargs: dict = {}
    if need_twotime:
        logger.info(
            f"forcing cropping to use only valid pixels: "
            f"{crop_ratio_threshold=:.2f} -> 1.0"
        )
        crop_ratio_threshold = 1.0
        qpm_kwargs.update(dq_selection=dq_selection, flag_sort=True)

    qpm = create_qpm(qmap, device, crop_ratio_threshold, **qpm_kwargs)

    if verbose:
        qpm.describe()
        logger.info(f"device: {device}")

    # --- shared build_run_args kwargs (save_results excluded: set per call site) ---
    shared = dict(
        qpm=qpm, device=device, verbose=verbose, num_loaders=num_loaders,
        meta_fname=meta_fname,
        qmap=qmap, output=output, overwrite=overwrite, prefix=prefix,
        batch_size=batch_size, normalize_frame=normalize_frame,
        num_partial_g2=num_partial_g2, max_memory=max_memory,
        save_G2=save_G2, bin_time_s=bin_time_s, smooth=smooth,
    )

    if analysis_type != "Both":
        common_dset_kwargs = {
            "device": device,
            "mask_crop": qpm.mask_crop,
            "avg_frame": avg_frame,
            "begin_frame": begin_frame,
            "end_frame": end_frame,
            "stride_frame": stride_frame,
        }
        if analysis_type == "Multitau":
            common_dset_kwargs["bin_time_s"] = bin_time_s
            common_dset_kwargs["run_config_path"] = run_config_path

        raw = load_raw_list(raw)
        jobs = build_jobs(
            raw, num_segments, begin_frame, suffix, common_dset_kwargs,
            label=f"{analysis_type.lower()} ",
        )
        run_args = _build_run_args(
            analysis_type, analysis_kwargs=analysis_kwargs,
            save_results=save_results, **shared,
        )
        run_jobs(jobs, **run_args)
        return

    # --- "Both": run multitau then twotime per raw file, merge into one result ---
    raw = load_raw_list(raw)

    base_dset_kwargs = {
        "device": device,
        "mask_crop": qpm.mask_crop,
        "avg_frame": avg_frame,
        "begin_frame": begin_frame,
        "end_frame": end_frame,
        "stride_frame": stride_frame,
    }
    multitau_dset_kwargs = {
        **base_dset_kwargs,
        "bin_time_s": bin_time_s,
        "run_config_path": run_config_path,
    }

    multitau_analysis_kwargs = {**analysis_kwargs, "analysis_type": "Multitau"}
    twotime_analysis_kwargs = {**analysis_kwargs, "analysis_type": "Twotime"}

    mt_run_args = _build_run_args(
        "Multitau", analysis_kwargs=multitau_analysis_kwargs, **shared,
        save_results=False,
    )
    tt_run_args = _build_run_args(
        "Twotime", analysis_kwargs=twotime_analysis_kwargs, **shared,
        save_results=False, skip_scattering=True,
    )

    n_files = len(raw)
    for file_idx, raw_file in enumerate(raw, start=1):
        t_file_start = time.perf_counter()
        mt_jobs = build_jobs(
            [raw_file], num_segments, begin_frame, suffix,
            multitau_dset_kwargs, label="multitau ",
        )
        tt_jobs = build_jobs(
            [raw_file], num_segments, begin_frame, suffix,
            base_dset_kwargs, label="twotime ",
        )

        rf_kwargs_m, payloads_m = run_jobs(mt_jobs, **mt_run_args)
        rf_kwargs_t, payloads_t = run_jobs(tt_jobs, **tt_run_args)

        rf_kwargs_m.update(rf_kwargs_t)
        try:
            with XpcsResult(**rf_kwargs_m) as result_file:
                for item in payloads_m:
                    result_file.append(item)
                for item in payloads_t:
                    result_file.append(item)
        except Exception as e:
            raise exc.ResultSavingError from e

        elapsed = time.perf_counter() - t_file_start
        log_job_status(file_idx, n_files, elapsed, f"saved: {result_file.fname}")

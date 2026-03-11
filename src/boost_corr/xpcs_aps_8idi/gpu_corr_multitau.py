import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

import boost_corr.xpcs_aps_8idi.exceptions as exc

from .. import MultitauCorrelator
from ..help_functions import get_device
from .dataset import create_dataset
from .xpcs_qpartitionmap import XpcsQPartitionMap
from .xpcs_result import XpcsResult

logger = logging.getLogger(__name__)


def solve_multitau(*args: Any, **kwargs: Any) -> Union[str, None]:
    num_rawfiles = len(kwargs["raw"])
    num_segments = kwargs["num_segments"]

    kwargs_record = kwargs.copy()
    kwargs_record["analysis_type"] = "multitau"

    if num_segments > 1:
        all_rawfiles = kwargs["raw"].copy()  # force copy
        for raw in all_rawfiles:
            kwargs["raw"] = [raw]
            solve_multitau_batch(*args, analysis_kwargs=kwargs_record, **kwargs)
        return
    else:
        # no segments
        if num_rawfiles == 1:
            kwargs["raw"] = kwargs["raw"][0]
            # return the result file name
            return solve_multitau_single(*args, analysis_kwargs=kwargs_record, **kwargs)
        else:
            solve_multitau_batch(*args, analysis_kwargs=kwargs_record, **kwargs)


def solve_multitau_single(
    qmap: Union[str, Path] = None,
    raw: Union[str, Path] = None,
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
    analysis_kwargs: Optional[dict] = None,
    save_results: bool = True,
    num_partial_g2: int = 0,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    bin_time_s: float = 1e-6,
    run_config_path=None,
    max_memory: float = 36.0,
    meta_fname: Optional[str] = None,
    **kwargs: Any,
) -> Union[str, None]:
    log_level = logging.INFO if verbose else logging.ERROR
    logger.setLevel(log_level)

    device = get_device(gpu_id)

    # create qpartitionmap
    try:
        qpm = XpcsQPartitionMap(qmap, device=device, crop_ratio_threshold=crop_ratio_threshold)
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
            bin_time_s=bin_time_s,
            run_config_path=run_config_path,
        )
    except Exception as e:
        raise exc.DatasetError from e

    # in some detectors/configurations, the qmap is rotated
    qpm.update_rotation(dset.det_size)

    try:
        xb = MultitauCorrelator(
            dset.det_size,
            frame_num=dset.frame_num,
            queue_size=batch_size,  # batch_size is the minimal value
            auto_queue=True,
            device=device,
            mask_crop=qpm.mask_crop,
            normalize_frame=normalize_frame,
            qpm=qpm,
            num_partial_g2=num_partial_g2,
            max_memory=max_memory,
        )
    except Exception as e:
        raise exc.CorrelatorError from e

    if verbose:
        dset.describe()
        xb.describe()
        logger.info("correlation solver created.")

    t_start = time.perf_counter()
    try:
        xb.process_dataset(dset, verbose=verbose, use_loader=use_loader, num_workers=num_loaders)
    except Exception as e:
        raise exc.ProcessingError from e

    t_end = time.perf_counter()
    t_diff = t_end - t_start
    frequency = dset.frame_num / t_diff
    logger.info(f"correlation finished in {t_diff:.2f}s." + f" frequency = {frequency:.2f} Hz")

    t_start = time.perf_counter()
    try:
        output_scattering, output_multitau = xb.get_results()
        norm_scattering = qpm.normalize_scattering(output_scattering)
        norm_multitau = qpm.normalize_multitau(output_multitau, save_G2=save_G2)
        part_multitau = xb.get_partial_g2()
    except Exception as e:
        raise exc.PostProcessingError from e

    t_end = time.perf_counter()
    logger.info("normalization finished in %.3fs" % (t_end - t_start))

    if save_results:
        try:
            with XpcsResult(
                raw_fname=raw,
                qmap_fname=qmap,
                output_dir=output,
                meta_fname=meta_fname,
                overwrite=overwrite,
                multitau_config=analysis_kwargs,
                prefix=prefix,
                suffix=suffix,
            ) as result_file:
                result_file.append(norm_scattering)
                result_file.append(norm_multitau)
                result_file.append(part_multitau)
                if dset.dataset_type == "Timepix4Dataset":
                    result_file.correct_t0_for_timepix4(bin_time_s)
            logger.info("multitau analysis finished")
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
            "multitau_config": analysis_kwargs,
            "prefix": prefix,
            "suffix": suffix,
        }
        return result_file_kwargs, (norm_scattering, norm_multitau)


def solve_multitau_batch(
    qmap: Union[str, Path] = None,
    raw: list[str] = None,
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
    analysis_kwargs: Optional[dict] = None,
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
    log_level = logging.INFO if verbose else logging.ERROR
    logger.setLevel(log_level)

    device = get_device(gpu_id)

    # create qpartitionmap
    try:
        qpm = XpcsQPartitionMap(qmap, device=device, crop_ratio_threshold=crop_ratio_threshold)
    except Exception as e:
        raise exc.QMapError from e

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

    if num_segments > 1:
        logger.info(f"multiple segments process for a single file, num_segments: {num_segments}")
        assert len(raw) == 1, "multiple segments process only supports one raw file"
        assert stride_frame == 1, "multiple segments process only supports stride frame = 1"
        assert avg_frame == 1, "multiple segments process only supports avg frame = 1"

        dset, use_loader = create_dataset(raw[0], **common_dset_kwargs)
        total_frame_num = dset.frame_num
        assert total_frame_num % num_segments == 0, "total frame number must be divisible by num_segments"
        segment_frame_num = total_frame_num // num_segments
        assert segment_frame_num >= 1, "segment frame number must be >= 1"

        dset_kwargs_list = []
        suffix_width = len(str(num_segments - 1))  # start with 0
        for n in range(num_segments):
            entry = common_dset_kwargs.copy()
            entry["begin_frame"] = begin_frame + n * segment_frame_num
            entry["end_frame"] = begin_frame + (n + 1) * segment_frame_num
            raw_fname = raw[0]
            _suffix = f"segment_{n:0{suffix_width}d}"
            if suffix is not None:
                _suffix = f"{suffix}_{_suffix}"
            dset_kwargs_list.append([raw_fname, entry, _suffix])

    else:
        logger.info(f"single segments process for multiple files, num_segments: {num_segments}")
        dset_kwargs_list = []
        for raw_fname in raw:
            entry = common_dset_kwargs.copy()
            dset_kwargs_list.append([raw_fname, entry, suffix])

    correlator = None
    for raw_fname, dset_kwargs, _suffix in dset_kwargs_list:
        try:
            dset, use_loader = create_dataset(raw_fname, **dset_kwargs)
        except Exception as e:
            raise exc.DatasetError from e
        # in some detectors/configurations, the qmap is rotated
        qpm.update_rotation(dset.det_size)

        if correlator is None:
            try:
                correlator = MultitauCorrelator(
                    dset.det_size,
                    frame_num=dset.frame_num,
                    queue_size=batch_size,  # batch_size is the minimal value
                    auto_queue=True,
                    device=device,
                    mask_crop=qpm.mask_crop,
                    normalize_frame=normalize_frame,
                    qpm=qpm,
                    num_partial_g2=num_partial_g2,
                    max_memory=max_memory,
                )
            except Exception as e:
                raise exc.CorrelatorError from e
        else:
            logger.info("reset correlator")
            correlator.reset()

        if verbose:
            dset.describe()
            correlator.describe()
            logger.info("correlation solver created.")

        t_start = time.perf_counter()
        try:
            correlator.process_dataset(dset, verbose=verbose, use_loader=use_loader, num_workers=num_loaders)
        except Exception as e:
            raise exc.ProcessingError from e

        t_end = time.perf_counter()
        t_diff = t_end - t_start
        frequency = dset.frame_num / t_diff
        logger.info(f"correlation finished in {t_diff:.2f}s." + f" frequency = {frequency:.2f} Hz")

        t_start = time.perf_counter()
        try:
            output_scattering, output_multitau = correlator.get_results()
            norm_scattering = qpm.normalize_scattering(output_scattering)
            norm_multitau = qpm.normalize_multitau(output_multitau, save_G2=save_G2)
            part_multitau = correlator.get_partial_g2()
        except Exception as e:
            raise exc.PostProcessingError from e

        t_end = time.perf_counter()
        logger.info("normalization finished in %.3fs" % (t_end - t_start))

        try:
            with XpcsResult(
                raw_fname=raw_fname,
                qmap_fname=qmap,
                output_dir=output,
                meta_fname=meta_fname,
                overwrite=overwrite,
                multitau_config=analysis_kwargs,
                prefix=prefix,
                suffix=_suffix,
            ) as result_file:
                result_file.append(norm_scattering)
                result_file.append(norm_multitau)
                result_file.append(part_multitau)
                if dset.dataset_type == "Timepix4Dataset":
                    result_file.correct_t0_for_timepix4(bin_time_s)
            logger.info("multitau analysis finished")
            # return result_file.fname
        except Exception as e:
            raise exc.ResultSavingError from e

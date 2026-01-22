import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

from .. import MultitauCorrelator
from .dataset import create_dataset
from .xpcs_qpartitionmap import XpcsQPartitionMap
from .xpcs_result import XpcsResult

logger = logging.getLogger(__name__)


def solve_multitau(*args: Any, **kwargs: Any) -> Union[str, None]:
    kwargs_record = kwargs.copy()
    kwargs_record["analysis_type"] = "multitau"
    return solve_multitau_base(*args, analysis_kwargs=kwargs_record, **kwargs)


def solve_multitau_base(
    qmap: Optional[Union[str, Path]] = None,
    raw: Optional[Union[str, Path]] = None,
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
    **kwargs: Any,
) -> Union[str, None]:
    log_level = logging.INFO if verbose else logging.ERROR
    logger.setLevel(log_level)

    device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"

    # create qpartitionmap
    qpm = XpcsQPartitionMap(
        qmap, device=device, crop_ratio_threshold=crop_ratio_threshold
    )

    if verbose:
        qpm.describe()
        logger.info(f"device: {device}")

    # create dataset
    dset, use_loader = create_dataset(
        raw,
        device,
        mask_crop=qpm.mask_crop,
        avg_frame=avg_frame,
        begin_frame=begin_frame,
        end_frame=end_frame,
        stride_frame=stride_frame,
        bin_time_s=bin_time_s,
        run_config_path=run_config_path,
    )

    # in some detectors/configurations, the qmap is rotated
    qpm.update_rotation(dset.det_size)

    # determine the metadata path
    # dirname(FILES_IN_CURRENT_FOLDER) gives empty string
    meta_dir = os.path.dirname(os.path.abspath(raw))

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
    )

    if verbose:
        dset.describe()
        xb.describe()
        logger.info("correlation solver created.")

    t_start = time.perf_counter()
    xb.process_dataset(
        dset, verbose=verbose, use_loader=use_loader, num_workers=num_loaders
    )
    t_end = time.perf_counter()
    t_diff = t_end - t_start
    frequency = dset.frame_num / t_diff
    logger.info(
        f"correlation finished in {t_diff:.2f}s." + f" frequency = {frequency:.2f} Hz"
    )

    t_start = time.perf_counter()
    output_scattering, output_multitau = xb.get_results()
    norm_scattering = qpm.normalize_scattering(output_scattering)
    norm_multitau = qpm.normalize_multitau(output_multitau, save_G2=save_G2)
    part_multitau = xb.get_partial_g2()
    t_end = time.perf_counter()
    logger.info("normalization finished in %.3fs" % (t_end - t_start))

    if save_results:
        with XpcsResult(
            meta_dir,
            qmap,
            output,
            overwrite=overwrite,
            multitau_config=analysis_kwargs,
            rawdata_path=os.path.realpath(raw),
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
    else:
        result_file_kwargs = {
            "meta_dir": meta_dir,
            "qmap_fname": qmap,
            "output_dir": output,
            "overwrite": overwrite,
            "multitau_config": analysis_kwargs,
        }
        return result_file_kwargs, (norm_scattering, norm_multitau)

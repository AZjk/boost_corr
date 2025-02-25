import logging
import os

import time
from pathlib import Path
from typing import Any, Optional, Union

from .. import TwotimeCorrelator
from .dataset import create_dataset
from .xpcs_qpartitionmap import XpcsQPartitionMap
from .xpcs_result import XpcsResult

logger = logging.getLogger(__name__)


def solve_twotime(*args: Any, **kwargs: Any) -> Union[str, None]:
    kwargs_record = kwargs.copy() 
    kwargs_record["analysis_type"] = 'twotime'
    return solve_twotime_base(*args, analysis_kwargs=kwargs_record, **kwargs)


def solve_twotime_base(
    qmap: Optional[Union[str, Path]] = None,
    raw: Optional[Union[str, Path]] = None,
    output: str = "cluster_results",
    batch_size: int = 8,
    gpu_id: int = 0,
    verbose: bool = False,
    masked_ratio_threshold: float = 0.85,
    num_loaders: int = 16,
    begin_frame: int = 0,
    end_frame: int = -1,
    avg_frame: int = 1,
    stride_frame: int = 1,
    overwrite: bool = False,
    # save_G2: bool = False,
    dq_selection: Optional[Union[str, Path]] = None,
    smooth: str = 'sqmap',
    analysis_kwargs: Optional[dict] = None,
    **kwargs):

    log_level = logging.INFO if verbose else logging.ERROR
    logger.setLevel(log_level)
    device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"

    # create qpartitionmap
    qpm = XpcsQPartitionMap(qmap, device=device, flag_sort=True,
                            masked_ratio_threshold=masked_ratio_threshold,
                            dq_selection=dq_selection)
    if verbose:
        qpm.describe()
        logger.info(f"device: {device}")

    logger.setLevel(log_level)

    # create dataset
    dset, use_loader = create_dataset(raw, device,
                                      mask_crop=qpm.mask_crop,
                                      avg_frame=avg_frame,
                                      begin_frame=begin_frame,
                                      end_frame=end_frame,
                                      stride_frame=stride_frame)

    # in some detectors/configurations, the qmap is rotated
    qpm.update_rotation(dset.det_size)
    # determine the metadata path
    # dirname(FILES_IN_CURRENT_FOLDER) gives empty string
    meta_dir = os.path.dirname(os.path.abspath(raw))

    twotime_correlator = TwotimeCorrelator(
                           qpm.qinfo,
                           frame_num=dset.frame_num,
                           det_size=dset.det_size,
                           method='normal',
                           mask_crop=qpm.mask_crop,
                           window=1024,
                           device=device)

    logger.info("correlation solver created.")

    t_start = time.perf_counter()
    twotime_correlator.process_dataset(
        dset, verbose=verbose, use_loader=use_loader, num_workers=num_loaders)
    twotime_correlator.post_processing(smooth_method=smooth)
    t_end = time.perf_counter()
    t_diff = t_end - t_start
    frequency = dset.frame_num / t_diff
    logger.info(f"correlation finished in {t_diff:.2f}s." +
                f" frequency = {frequency:.2f} Hz")

    t_start = time.perf_counter()
    saxs = twotime_correlator.get_saxs()
    norm_data = qpm.normalize_saxs(saxs)
    t_end = time.perf_counter()
    logger.info("normalization finished in %.3fs" % (t_end - t_start))

    # saving results to file
    with XpcsResult(meta_dir, qmap, output, overwrite=overwrite,
                    twotime_config=analysis_kwargs) as result_file:
        result_file.save(norm_data)
        for info in twotime_correlator.get_twotime_result():
            result_file.save_incremental_results(info)
    logger.info(f"twotime analysis finished")
    return result_file.fname

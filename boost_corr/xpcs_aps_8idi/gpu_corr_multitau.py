from typing import Optional, Union, Any
from pathlib import Path
import time
import logging
import os
from .. import MultitauCorrelator
from .xpcs_result import XpcsResult
from .xpcs_qpartitionmap import XpcsQPartitionMap
from .dataset import create_dataset


logger = logging.getLogger(__name__)


def solve_multitau(
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
    save_G2: bool = False,
    **kwargs: Any
) -> Union[str, None]:

    log_level = logging.INFO if verbose else logging.ERROR
    logger.setLevel(log_level)

    device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"

    # create qpartitionmap
    qpm = XpcsQPartitionMap(qmap, device=device,
                            masked_ratio_threshold=masked_ratio_threshold)

    if verbose:
        qpm.describe()
        logger.info(f"device: {device}")

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

    if not os.path.isdir(output):
        os.makedirs(output)

    xb = MultitauCorrelator(
        dset.det_size,
        frame_num=dset.frame_num,
        queue_size=batch_size,  # batch_size is the minimal value
        auto_queue=True,
        device=device,
        mask_crop=qpm.mask_crop)

    if verbose:
        dset.describe()
        xb.describe()
        logger.info("correlation solver created.")

    t_start = time.perf_counter()
    xb.process_dataset(dset, verbose=verbose, use_loader=use_loader,
                       num_workers=num_loaders)
    t_end = time.perf_counter()
    t_diff = t_end - t_start
    frequency = dset.frame_num / t_diff
    logger.info(f"correlation finished in {t_diff:.2f}s." +
                f" frequency = {frequency:.2f} Hz")

    t_start = time.perf_counter()
    result = xb.get_results()
    result = qpm.normalize_data(result, save_G2=save_G2)
    t_end = time.perf_counter()
    logger.info("normalization finished in %.3fs" % (t_end - t_start))

    result_file = XpcsResult(meta_dir, qmap, output, avg_frame=avg_frame,
                             stride_frame=stride_frame, overwrite=overwrite)
    result_file.safe_save(result)
    logger.info(f"analysis results saved as {result_file.fname}")

    return result_file.fname

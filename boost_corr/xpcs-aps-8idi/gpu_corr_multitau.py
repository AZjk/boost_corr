import time
import logging
import os
from .. import MultitauCorrelator
from imm_handler import ImmDataset
from rigaku_handler import RigakuDataset
from hdf_handler import HdfDataset
from xpcs_result import XpcsResult
import magic
from xpcs_qpartitionmap import XpcsQPartitionMap


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s T+%(relativeCreated)05dms [%(filename)s]: %(message)s",
    datefmt="%m-%d %H:%M:%S")


logger = logging.getLogger(__name__)


def solve_multitau(qmap=None,
                   raw=None,
                   output="cluster_results",
                   batch_size=8,
                   gpu_id=0,
                   verbose=False,
                   masked_ratio_threshold=0.75,
                   use_loader=True,
                   begin_frame=3,
                   end_frame=-1,
                   avg_frame=7,
                   stride_frame=5,
                   overwrite=False,
                   **kwargs):

    log_level = logging.ERROR
    if verbose:
        log_level = logging.INFO

    logger.setLevel(log_level)

    meta_dir = os.path.dirname(raw)

    # log task info
    logger.info(f"meta_dir: {meta_dir}")
    logger.info(f"qmap: {qmap}")
    logger.info(f"output: {output}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"gpu_id: {gpu_id}")

    if not os.path.isdir(output):
        # os.mkdir(output)
        os.makedirs(output)

    if gpu_id >= 0:
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    logger.info(f"device: {device}")

    qpm = XpcsQPartitionMap(qmap, device=device)
    logger.info("QPartitionMap instance created.")
    logger.info(f"masked area: {qpm.masked_pixels}")
    logger.info(f"masked area ratio: {qpm.masked_ratio:0.3f}")
    result_file = XpcsResult(meta_dir, qmap, output, overwrite=overwrite)

    # determine whether to use mask or not based on the mask"s ratio
    if qpm.masked_ratio > masked_ratio_threshold:
        mask_crop = None
    else:
        mask_crop = qpm.get_mask_crop()
        logger.info(f"masked_ratio is too low. will crop the raw input.")

    ext = os.path.splitext(raw)[-1]

    # use_loader is set for HDF files, it can use multiple processes to read
    # large HDF file;
    use_loader = False
    if ext == ".bin":
        dataset_method = RigakuDataset
        use_loader = False
        batch_size = 1024
    elif ext in [".imm", ".h5", ".hdf"]:
        ftype = magic.from_file(raw)
        if ftype == "Hierarchical Data Format (version 5) data":
            dataset_method = HdfDataset
            use_loader = True
            batch_size = 1
        else:
            dataset_method = ImmDataset
            use_loader = False
            batch_size = 256

    dset = dataset_method(raw,
                          batch_size=batch_size,
                          device=device,
                          mask_crop=mask_crop,
                          avg_frame=avg_frame,
                          begin_frame=begin_frame,
                          end_frame=end_frame,
                          stride_frame=stride_frame)

    flag_rot = qpm.update_rotation(dset.det_size)
    if flag_rot:
        dset.update_mask_crop(qpm.get_mask_crop())

    xb = MultitauCorrelator(
        dset.det_size,
        frame_num=dset.frame_num,
        queue_size=batch_size,  # batch_size is the minimal value
        auto_queue=True,
        device_type=device,
        mask_crop=mask_crop,
        avg_frame=avg_frame)

    if verbose:
        dset.describe()
        xb.describe()

    logger.info("correlation solver created.")

    t_start = time.perf_counter()
    xb.process_dataset(dset, verbose=verbose, use_loader=use_loader)
    t_end = time.perf_counter()
    t_diff = t_end - t_start
    frequency = dset.frame_num / t_diff
    logger.info(f"correlation finished in {t_diff:.2f}s." +
                f" frequency = {frequency:.2f} Hz")

    t_start = time.perf_counter()
    result = xb.get_results()
    result = qpm.normalize_data(result)
    result_file.save(result)
    t_end = time.perf_counter()
    logger.info("normalization finished in %.3fs" % (t_end - t_start))
    logger.info(f"analysis results exported to {output}")

    return result_file.fname

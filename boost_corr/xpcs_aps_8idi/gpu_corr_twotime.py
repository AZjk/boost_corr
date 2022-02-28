import logging
import os
from .imm_handler import ImmDataset
from .rigaku_handler import RigakuDataset
from .hdf_handler import HdfDataset
from .xpcs_result import XpcsResult
import magic
from torch.utils.data import DataLoader
from .xpcs_qpartitionmap import XpcsQPartitionMap
from .. import TwotimeCorrelator
# from torch.profiler import profile, record_function, ProfilerActivity


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s T+%(relativeCreated)05dms [%(filename)s]: %(message)s",
    datefmt="%m-%d %H:%M:%S")

logger = logging.getLogger(__name__)


def solve_twotime(qmap=None,
                  raw=None,
                  output="cluster_results",
                  batch_size=1,
                  gpu_id=0,
                  verbose=False,
                  begin_frame=1,
                  end_frame=-1,
                  avg_frame=1,
                  stride_frame=1,
                  dq_selection=None,
                  smooth='sqmap',
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

    qpm = XpcsQPartitionMap(qmap, flag_fix=True, flag_sort=True,
                            device=device, dq_selection=dq_selection)

    logger.info("QPartitionMap instance created.")
    logger.info(f"masked area: {qpm.masked_pixels}")
    logger.info(f"masked area ratio: {qpm.masked_ratio:0.3f}")
    result_file = XpcsResult(meta_dir, qmap, output, analysis_type='Twotime')

    # determine whether to use mask or not based on the mask"s ratio
    # if qpm.masked_ratio > masked_ratio_threshold:
    #     mask_crop = None
    if True:
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

    dset = dataset_method(raw, batch_size=batch_size, device=device,
                          mask_crop=mask_crop, avg_frame=avg_frame,
                          begin_frame=begin_frame, end_frame=end_frame,
                          stride_frame=stride_frame)
    if verbose:
        dset.describe()

    flag_rot = qpm.update_rotation(dset.det_size)
    if flag_rot:
        dset.update_mask_crop(qpm.get_mask_crop())

    tt = TwotimeCorrelator(qpm.qinfo,
                           frame_num=dset.frame_num,
                           det_size=dset.det_size,
                           method='normal',
                           mask_crop=mask_crop,
                           window=1024,
                           device=device)

    logger.info("correlation solver created.")

    dl = DataLoader(dset)

    for x in dl:
        # print(x.dtype)
        tt.process(x)

    logger.info(f"smooth method used: {smooth}")
    tt.post_processing(smooth_method=smooth)

    for info in tt.get_twotime_result():
        result_file.save(info, mode='raw', compression='gzip')

    saxs = tt.get_saxs()
    norm_data = qpm.normalize_saxs(saxs)
    result_file.save(norm_data)

    logger.info(f"analysis results exported to {output}")

    return result_file.fname

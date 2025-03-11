"""Module for GPU correlation solver.
This module contains functions for checking metadata, extracting raw metadata, and solving GPU correlation.
"""

import logging
import os
import queue
import time
import traceback
import uuid

import h5py
import magic
import torch
from PyQt5 import QtCore
from torch.utils.data import DataLoader
from xpcs_boost import XpcsBoost as XB

# from torch.profiler import profile, record_function, ProfilerActivity
# from .aps_8idi import key as hdf_key
# from .hdf_handler import HdfDataset
from boost_corr.xpcs_aps_8idi.dataset.hdf_handler import HdfDataset

# from .imm_handler import ImmDataset
from boost_corr.xpcs_aps_8idi.dataset.imm_handler import ImmDataset

# from .rigaku_handler import RigakuDataset
from boost_corr.xpcs_aps_8idi.dataset.rigaku_handler import RigakuDataset

# from .xpcs_metadata import XpcsMetaData
from boost_corr.xpcs_aps_8idi.service.xpcs_metadata import XpcsMetaData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s T+%(relativeCreated)05dms [%(filename)s]: %(message)s",
    datefmt="%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def is_metadata(fname: str):
    """TODO: Add docstring for is_metadata.

    Parameters:
        fname (str): The filename to check.

    Returns:
        bool: True if the file is metadata, otherwise False.
    """
    try:
        with h5py.File(fname, "r") as f:
            return "/hdf_metadata_version" in f
    except Exception:
        return False


def get_raw_meta(raw_folder: str):
    """TODO: Add docstring for get_raw_meta.

    Parameters:
        raw_folder (str): The folder containing raw data.

    Returns:
        tuple: A tuple containing raw and metadata file information.
    """
    fs_pure = os.listdir(raw_folder)
    fs = [os.path.join(raw_folder, x) for x in fs_pure]
    raw = None
    meta = None
    if len(fs) != 3:
        return None, None
    for f in fs:
        if f.endswith(".batchinfo"):
            continue
        elif is_metadata(f):
            meta = f
        else:
            raw = f
    return (raw, meta)


def solve(
    qmap=None,
    raw_folder=None,
    output="cluster_results",
    batch_size=8,
    gpu_id=0,
    verbose=True,
    masked_ratio_threshold=0.75,
    data_begin_todo=1,
    data_end_todo=-1,
    avg_frames=1,
    stride_frames=1,
    signal_progress=None,
    config=None,
    num_worker=8,
    max_memory=12.0,
    **kwargs,
):
    """TODO: Add docstring for solve.

    Parameters:
        qmap: Qmap file or identifier.
        raw_folder: Folder with raw data files.
        output (str): Output directory for results.
        batch_size (int): Batch size for processing.
        gpu_id (int): GPU identifier. Use -1 for CPU.
        verbose (bool): Verbose flag.
        masked_ratio_threshold (float): Threshold for mask ratio.
        data_begin_todo: Starting data index.
        data_end_todo: Ending data index.
        avg_frames (int): Number of frames to average.
        stride_frames (int): Frame stride.
        signal_progress: Signal for progress updates.
        config (dict): Configuration dictionary.
        num_worker (int): Number of worker threads for PyTorch.
        max_memory (float): Maximum memory allocation in GB.
        **kwargs: Additional parameters.

    Returns:
        None
    """
    torch.set_num_threads(num_worker)
    raw, meta = get_raw_meta(raw_folder)
    meta_dir = raw_folder

    job_info = {
        "job_status": "submitted",
        "meta_dir": meta_dir,
        "qmap": qmap,
        "output": output,
        "batch_size": batch_size,
        "gpu_id": gpu_id,
    }

    if not os.path.isdir(output):
        # os.mkdir(output)
        os.makedirs(output)

    if gpu_id >= 0:
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    job_info["device"] = device

    meta = XpcsMetaData(meta_dir, qmap, output, exclude_raw=raw)
    job_info["masked area ratio"] = meta.masked_ratio
    job_info["masked area"] = meta.masked_area

    # determine whether to use mask or not based on the mask's ratio
    if meta.masked_ratio > masked_ratio_threshold:
        mask = None
        job_info["crop input"] = False
    else:
        mask = meta.mask
        job_info["crop input"] = True

    ext = os.path.splitext(raw)[-1]
    # use_loader is set for HDF files, it can use multiple processes to read
    # large HDF file;
    if ext == ".bin":
        dataset_method = RigakuDataset
    elif ext in [".imm", ".h5", ".hdf"]:
        ftype = magic.from_file(raw)
        if ftype == "Hierarchical Data Format (version 5) data":
            dataset_method = HdfDataset
        else:
            dataset_method = ImmDataset

    dset = dataset_method(raw, batch_size=batch_size, device=device, mask=mask)
    config["frame_num"] = dset.frame_num

    job_info["crop input"] = True
    job_info.update(dset.get_description())

    meta.update_rotation(dset.det_size)
    xb = XB(
        dset.det_size,
        frame_num=dset.frame_num,
        queue_size=batch_size,  # batch_size is the minimal value
        auto_queue=True,
        device_type=device,
        max_memory=max_memory,
        mask=mask,
    )
    # xb.describe()

    t_start = time.perf_counter()

    pin_memory = device != "cpu"
    dl = DataLoader(
        dset,
        batch_size=None,
        pin_memory=pin_memory,
        num_workers=num_worker,
        prefetch_factor=2,
    )

    total = len(dset)
    curr = 0
    p0 = -1
    for x in dl:
        curr += 1
        xb.process(x)
        p = round((curr * 100) / total, 1)
        if signal_progress is not None and p > (p0 + 0.2):
            signal_progress.emit((0, 0))
            config["progress"] = p
            p0 = p

    xb.post_process()
    del dl
    t_end = time.perf_counter()
    t_diff = t_end - t_start
    frequency = dset.frame_num / t_diff

    job_info["correlation time (s)"] = t_diff
    job_info["correlation frequency"] = frequency

    t_start = time.perf_counter()
    result = xb.get_results(meta)
    t_end = time.perf_counter()
    job_info["normalization time (s)"] = t_end - t_start
    meta.append_analysis_result(result)
    job_info["job_status"] = "finished"
    config["time"] = str(round(t_diff, 1)) + "s/" + str(round(frequency)) + "Hz"
    del xb
    del meta

    return


class WorkerSignal(QtCore.QObject):
    """TODO: Add docstring for WorkerSignal class.

    This class defines the signals used by GPU solver workers.
    """

    progress = QtCore.pyqtSignal(tuple)
    values = QtCore.pyqtSignal(tuple)
    status = QtCore.pyqtSignal(tuple)


class GPUSolverWorker(QtCore.QRunnable):
    """TODO: Add docstring for GPUSolverWorker class.

    This class represents a worker for GPU solving tasks.
    """

    def __init__(self, jid=0, raw_folder=None, **kwargs):
        """TODO: Add docstring for GPUSolverWorker.__init__.

        Parameters:
            jid: Job id (unique identifier for the job).
            raw_folder: Folder containing raw data files.
            **kwargs: Additional parameters.
        """
        super(GPUSolverWorker, self).__init__()
        self.signals = WorkerSignal()
        if jid is None:
            self.jid = str(uuid.uuid4())
        else:
            self.jid = jid
        self.fname = os.path.basename(raw_folder)
        self.kwargs = kwargs
        self.kwargs["raw_folder"] = raw_folder
        self.config = {
            "frame_num": -1,
            "progress": 0,
            "status": "queued",
            "jid": jid,
            "fname": self.fname,
            "time": -1,
        }
        self.config.update(self.kwargs)

    def get_data(self):
        """TODO: Add docstring for GPUSolverWorker.get_data.

        Returns:
            list: A list containing key job configuration data.
        """
        result = []
        for key in [
            "jid",
            "priority",
            "gpu_id",
            "status",
            "frame_num",
            "progress",
            "time",
            "fname",
        ]:
            result.append(self.config.get(key))
        return result

    @QtCore.pyqtSlot()
    def run(self):
        """TODO: Add docstring for GPUSolverWorker.run.
        Process the solving task and update job status.
        """
        self.config["status"] = "running"
        try:
            solve(
                config=self.config, signal_progress=self.signals.progress, **self.kwargs
            )
            self.config["status"] = "finished"
        except Exception as err:
            self.config["status"] = "failed"
            print("job failed", err)
            traceback.print_exc()
        self.signals.status.emit((self.jid, "finished"))
        return


class GPUSolverProducer:
    """TODO: Add docstring for GPUSolverProducer class.

    This class manages the submission of GPU solver jobs.
    """

    def __init__(self) -> None:
        """TODO: Add docstring for GPUSolverProducer.__init__.
        Initializes the job queue and job history.
        """
        self.job_queue = queue.Queue()
        self.job_history = []

    def submit_job(self, new_kwargs):
        """TODO: Add docstring for submit_job method.
        Submits a new job to the job queue.

        Parameters:
            new_kwargs: Dictionary of job parameters.
        """
        self.job_queue.put(new_kwargs)
        self.job_history.append(new_kwargs)

    def get_data(self):
        """TODO: Add docstring for get_data method.
        Retrieves job data for all submitted jobs.

        Returns:
            list: List of job data.
        """
        result = []
        for key in ["jid", "status", "frame_num", "progress", "fname"]:
            result.append(self.config.get(key))
        return result


class GPUSolverConsumer(QtCore.QRunnable):
    """TODO: Add docstring for GPUSolverConsumer class.

    This class consumes jobs from the queue and processes them using a thread pool.
    """

    def __init__(self, thread_pool, job_queue) -> None:
        """TODO: Add docstring for GPUSolverConsumer.__init__.

        Parameters:
            thread_pool: The thread pool to run jobs.
            job_queue: Queue containing job parameters.
        """
        super(GPUSolverConsumer, self).__init__()
        self.thread_pool = thread_pool
        self.job_queue = job_queue

    def run(self):
        """TODO: Add docstring for GPUSolverConsumer.run.
        Continuously consumes and processes jobs from the job queue.
        """
        while True:
            print("before get")
            kwargs = self.job_queue.get()
            print("got:", kwargs)
            worker = GPUSolverWorker(**kwargs)
            self.thread_pool.start(worker)
            time.sleep(10)
            print("finish work")


if __name__ == "__main__":
    pass

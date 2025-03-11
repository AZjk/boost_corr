"""Module for GPU correlation server service in xpcs_aps_8idi.
This module provides classes and functions to manage GPU-based correlation jobs.
TODO: Add detailed documentation.
"""

import argparse
import logging
import os
import socket
import time
import traceback
from multiprocessing import Process, Queue, Value

import magic
import numpy as np
import torch
import zmq

from boost_corr.xpcs_aps_8idi.dataset.hdf_handler import HdfDataset
from boost_corr.xpcs_aps_8idi.dataset.imm_handler import ImmDataset
from boost_corr.xpcs_aps_8idi.dataset.rigaku_handler import RigakuDataset
from boost_corr.xpcs_aps_8idi.xpcs_boost import XpcsBoost as XB
from boost_corr.xpcs_aps_8idi.xpcs_qpartitionmap import XpcsQPartitionMap
from boost_corr.xpcs_aps_8idi.xpcs_result import XpcsResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(name)-12s |%(levelname)s| %(message)s",
    datefmt="%m-%d %H:%M:%S",
)


def get_system_information():
    """TODO: Add docstring for get_system_information.

    Returns:
        dict: System information (e.g., number of GPUs).
    """
    sys_info = {}
    sys_info["num_gpus"] = torch.cuda.device_count()
    # sys_info['num_cpus'] = psutil.cpu_count(logical=False)
    scale = 1024**3

    device = {}
    for n in range(sys_info["num_gpus"]):
        a = torch.cuda.get_device_properties("cuda:%d" % n)
        device[n] = {"name": a.name[7:], "total_memory": a.total_memory / scale}

    sys_info["device"] = device
    # cpu_ram = psutil.virtual_memory().total / scale
    # # set an upper limit to cpu_ram;
    # cpu_ram = min(36.0, cpu_ram)
    # device[-1] = {'name': 'cpu', 'total_memory': cpu_ram}

    return sys_info


class GPUWorker(Process):
    """TODO: Add docstring for GPUWorker class.

    This class represents a worker that processes GPU correlation jobs.
    """

    def __init__(self, device, worker_id, job_queue) -> None:
        """TODO: Add docstring for GPUWorker.__init__.

        Parameters:
            device: The processing device.
            worker_id: Unique identifier for the worker.
            job_queue: Queue of jobs.
        """
        super(GPUWorker, self).__init__()
        self.xb = None
        self.qpm = None
        self.job_queue = job_queue
        self.device = device
        self.worker_id = worker_id
        self.logger = logging.getLogger(f"[{self.worker_id}]")
        self.logger.info(f"worker initialized, {self.device}")

    def run(self):
        """TODO: Add docstring for GPUWorker.run.
        Process the assigned GPU job.
        """
        # print('worker started', self.worker_id)
        while True:
            try:
                kwargs = self.job_queue.get()
            except KeyboardInterrupt:
                self.logger.info("keyboard interrupt. quit now")
                return
            if kwargs["cmd"] == "quit":
                self.logger.info("receive quit command. quit now")
                return
            kwargs.pop("cmd", None)
            t_start = time.perf_counter()
            basename = os.path.basename(kwargs["raw"])

            try:
                self.process(**kwargs)
            except Exception:
                self.logger.error(f"job faild: {basename}")
                traceback.print_exc()
                self.xb = None
            else:
                t_diff = round(time.perf_counter() - t_start, 3)
                self.logger.info(f"job finished in [{t_diff}]s: {basename}")

    def process(
        self,
        qmap=None,
        raw=None,
        output="cluster_results",
        verbose=False,
        masked_ratio_threshold=0.75,
    ):
        """TODO: Add docstring for GPUWorker.process.

        Process the GPU job.
        """
        meta_dir = os.path.dirname(raw)
        if not os.path.isdir(output):
            os.makedirs(output)

        if self.qpm is None or self.qpm.fname != qmap:
            self.qpm = XpcsQPartitionMap(qmap, device=self.device)
            # reset xb
            self.xb = None

        result_file = XpcsResult(meta_dir, qmap, output, exclude_raw=raw)

        # determine whether to use mask or not based on the mask"s ratio
        if self.qpm.masked_ratio > masked_ratio_threshold:
            mask_crop = None
        else:
            mask_crop = self.qpm.get_mask_crop()

        ext = os.path.splitext(raw)[-1]

        # use_loader is set for HDF files, it can use multiple processes to
        #  read large HDF file;

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
                batch_size = 256
            else:
                dataset_method = ImmDataset
                use_loader = False
                batch_size = 256

        dset = dataset_method(
            raw, batch_size=batch_size, device=self.device, mask_crop=mask_crop
        )
        # dset.describe()

        flag_rot = self.qpm.update_rotation(dset.det_size)
        if flag_rot:
            dset.update_mask_crop(self.qpm.get_mask_crop())

        if self.xb is None or self.xb.frame_num != dset.frame_num:
            self.xb = XB(
                dset.det_size,
                frame_num=dset.frame_num,
                queue_size=batch_size,  # batch_size is the minimal value
                auto_queue=True,
                device_type=self.device,
                mask_crop=mask_crop,
            )
            # xb.describe()
        else:
            self.xb.reset()

        self.xb.process_dataset(dset, verbose=verbose, use_loader=use_loader)
        result = self.xb.get_results()
        result = self.qpm.normalize_data(result)
        result_file.save(result)
        return


class GPUProducer(Process):
    """TODO: Add docstring for GPUProducer class.

    This class produces GPU jobs and manages the job queue.
    """

    def __init__(self, port, job_queue, num_workers, num_jobs):
        """TODO: Add docstring for GPUProducer.__init__.

        Parameters:
            port: Port number to listen on.
            job_queue: Queue for jobs.
            num_workers: Number of worker processes.
            num_jobs: Total number of jobs.
        """
        super(GPUProducer, self).__init__()
        self.cid = "producer_00"
        self.logger = logging.getLogger(f"[{self.cid}]")
        self.job_queue = job_queue
        self.num_workers = num_workers
        self.num_jobs = num_jobs
        self.port = port

    def run(self):
        """TODO: Add docstring for GPUProducer.run.
        Start producing and submitting GPU jobs.
        """
        self.logger.info(f"producer initialized, listen {self.port}")
        context = zmq.Context()
        p_socket = context.socket(zmq.PAIR)
        p_socket.bind("tcp://*:%s" % self.port)
        while True:
            try:
                payload = p_socket.recv_json()
            except (KeyboardInterrupt, Exception):
                self.logger.info("keyboard interrupt. quit now")
                break
            if payload["cmd"] == "quit":
                self.logger.info("receive quit command. quit now")
                break

            elif payload["cmd"] in ["status", "Status"]:
                qsize = self.job_queue.qsize()
                status = f"Active|remain/total: {qsize}/{self.num_jobs.value}"
                p_socket.send_json({"status": f"{status}"})
            else:
                self.job_queue.put(payload)
                self.num_jobs.value += 1
        p_socket.close()
        context.term()


CACHE_SIZE = 1024


class GPUServer:
    """TODO: Add docstring for GPUServer class.

    This class sets up and runs the GPU correlation server.
    """

    def __init__(
        self,
        port=5555,
        output="/scratch/cluster_results",
        gpu_selection=None,
        worker_per_gpu=1,
    ) -> None:
        """TODO: Add docstring for GPUServer.__init__.

        Parameters:
            port (int): Port number for the server.
            output (str): Output directory for result files.
            gpu_selection (list): List of booleans indicating which GPUs to use.
            worker_per_gpu (int): Number of workers per GPU.
        """
        self.port = port
        self.ip_addr = str(socket.gethostbyname(socket.gethostname()))
        self.output_dir = output
        self.job_queue = Queue()
        self.num_jobs = Value("i", 0)
        sys_info = get_system_information()
        if gpu_selection is None:
            gpu_selection = [True] * sys_info["num_gpus"]
        num_workers = len(gpu_selection) * worker_per_gpu
        self.num_workers = num_workers

        # start workers
        self.p_pool = []
        for n in range(num_workers):
            gpu_id = n // worker_per_gpu
            if gpu_selection[gpu_id]:
                device = f"cuda:{gpu_id}"
                worker_id = sys_info["device"][gpu_id]["name"] + f"_{n}"
                worker = GPUWorker(device, worker_id, self.job_queue)
                worker.start()
                self.p_pool.append(worker)

        # start producers
        self.producer = GPUProducer(
            self.port, self.job_queue, num_workers, self.num_jobs
        )
        self.producer.start()
        self.p_pool.append(self.producer)

        self.job_info = np.zeros((CACHE_SIZE * 2, 4), dtype=np.float32)
        self.job_ptr = CACHE_SIZE

    def get_status(self, window):
        """TODO: Add docstring for GPUServer.get_status.

        Parameters:
            window: Status window size or context.

        Returns:
            tuple: (Flag, message, jobs information).
        """
        status = "Active: "
        num_jobs = self.num_jobs.value
        try:
            qsize = self.job_queue.qsize()
            num_done = num_jobs - qsize
            last_info = self.job_info[self.job_ptr - 1]
            rate = num_done - last_info[2]
            new_info = (num_jobs, qsize, num_done, rate)
            status += f"| remain/total: {num_jobs - num_done}/{num_jobs}"

            self.job_info[self.job_ptr] = np.array(new_info)
            self.job_ptr += 1

            if self.job_ptr == CACHE_SIZE * 2:
                self.job_info = np.roll(self.job_info, shift=-CACHE_SIZE, axis=0)
                self.job_ptr = CACHE_SIZE
            sl = slice(self.job_ptr - window, self.job_ptr)

            return True, status, self.job_info[sl]

        except Exception as e:
            return False, "Inactive: " + str(e), None

    def stop_server(self):
        """TODO: Add docstring for GPUServer.stop_server.
        Stop the server gracefully.
        """
        # empty the job queue
        try:
            while True:
                self.job_queue.get_nowait()
        except Exception:
            pass

        # put quit cmd in the queue
        for _ in range(self.num_workers):
            self.job_queue.put({"cmd": "quit"})

        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.connect(f"tcp://{self.ip_addr}:{self.port}")
        socket.send_json({"cmd": "quit"})
        for _n, x in enumerate(self.p_pool):
            x.join()

    def submit_jobs(self, flist):
        """TODO: Add docstring for GPUServer.submit_jobs.
        Submit jobs from the given file list.

        Parameters:
            flist: List of job file paths.
        """
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.connect(f"tcp://{self.ip_addr}:{self.port}")

        for f in flist:
            # logging.info(f'send: {f}')
            socket.send_json(
                {
                    "cmd": "multitau",
                    "raw": f,
                    "qmap": "/scratch/xpcs_data_raw/qmap/foster202110_qmap_RubberDonut_Lq0_S180_18_D18_18.h5",
                    "output": "/scratch/cluster_results",
                    "verbose": False,
                }
            )
            # socket.send_json({
            #     'cmd': 'multitau',
            #     'raw': f,
            #     'qmap': '/net/s8iddata/export/8-id-i/partitionMapLibrary/2021-3/tingxu202111_S360_D36_Lin_Rq1.h5',
            #     'output': '/scratch/cluster_results',
            #     'verbose': False})
            # time.sleep(0.001)

    def process_all_files(self, fname="/clhome/MQICHU/filelist2.txt"):
        """TODO: Add docstring for GPUServer.process_all_files.
        Process all files listed in the given file.

        Parameters:
            fname (str): Path to the file list.
        """
        flist = []
        with open(fname) as f:
            for line in f:
                flist.append(line[:-1])
        self.submit_jobs(flist)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    description = "XPCS-Boost Server"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT_DIR",
        type=str,
        required=False,
        default="cluster_results",
        help="""[default: cluster_results] the output directory
                                for the result file. If not exit, the program will
                                create this directory.""",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    # print(kwargs)
    gs = GPUServer()
    gs.process_all_files("/clhome/MQICHU/foster_imm.txt")

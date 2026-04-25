import logging
import os
import random
import signal
import time
from datetime import datetime
from getpass import getuser
from uuid import uuid4

import pynvml
from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)


class GPUScheduler:
    # /dev/shm is shared across all users on the host, so namespace by username.
    # Suffix (not nested) avoids needing a shared boost_corr parent dir.
    _USER_DIR = os.path.join("/dev/shm", f"boost_corr_{getuser()}")
    LOCK_DIR = os.path.join(_USER_DIR, "gpu_locks")
    QUEUE_DIR = os.path.join(_USER_DIR, "gpu_queue")

    def __init__(self, max_try=1000, sleep_duration=3, priority=5):
        self.max_try = max_try
        self.sleep_duration = sleep_duration
        self.priority = priority  # Lower number means higher priority
        self.lock_acquired = False
        self.lock_file = None
        self.gpu_id = None
        self.queue_file = None
        self.original_sigint_handler = None
        # Per-user dirs created lazily on instantiation, not at import.
        os.makedirs(self.LOCK_DIR, mode=0o700, exist_ok=True)
        os.makedirs(self.QUEUE_DIR, mode=0o700, exist_ok=True)

    def __enter__(self):
        pynvml.nvmlInit()
        try:
            self.num_gpus = pynvml.nvmlDeviceGetCount()
            if self.num_gpus == 0:
                raise RuntimeError("No NVIDIA GPUs detected.")

            # Register signal handler only after we know there's work to do.
            self.original_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._signal_handler)

            gpu_list = list(range(self.num_gpus))
            random.shuffle(gpu_list)

            # Place the job in the queue with priority
            self.queue_file = os.path.join(
                self.QUEUE_DIR, f"{self.priority}_{datetime.now().timestamp()}_{uuid4()}"
            )
            open(self.queue_file, "w").close()  # Create an empty file

            for _ in range(self.max_try):
                # Check if this job is at the front of the queue
                queue_files = sorted(os.listdir(self.QUEUE_DIR))
                if os.path.basename(self.queue_file) != queue_files[0]:
                    index = queue_files.index(os.path.basename(self.queue_file))
                    sleep_time = min(max(self.sleep_duration, 1 * index), 100)
                    logger.info(
                        f"Current position in the queue: {index + 1}/{len(queue_files)}, "
                        f"sleeping for {sleep_time:.1f} seconds..."
                    )
                    time.sleep(sleep_time)
                    continue

                # Try to acquire a GPU starting from the random GPU
                for gpu_id in gpu_list:
                    lock_path = os.path.join(self.LOCK_DIR, f"gpu_{gpu_id}.lock")
                    lock = FileLock(lock_path, timeout=0)
                    try:
                        lock.acquire()
                        self.lock_acquired = True
                        self.lock_file = lock
                        self.gpu_id = gpu_id
                        # Set the environment variable for the GPU
                        # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
                        logger.info(f"Allocated GPU {self.gpu_id}")
                        # Remove the queue file as we're proceeding
                        os.remove(self.queue_file)
                        return self  # Successfully acquired GPU
                    except Timeout:
                        continue  # GPU is in use, try next one

                # No GPUs were free, wait before retrying
                logger.info(
                    f"No GPUs are free at the moment. Retrying in {self.sleep_duration} seconds..."
                )
                time.sleep(self.sleep_duration)

            raise RuntimeError(f"Failed to acquire a GPU after {self.max_try} attempts.")
        except BaseException:
            # __exit__ won't run because __enter__ didn't return; clean up here
            # so a retry from the caller starts from a fresh global state.
            self._cleanup()
            if self.original_sigint_handler is not None:
                signal.signal(signal.SIGINT, self.original_sigint_handler)
                self.original_sigint_handler = None
            pynvml.nvmlShutdown()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore original signal handler
        if self.original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self.original_sigint_handler)
            self.original_sigint_handler = None

        # Release the GPU lock if acquired
        self._cleanup()

        # Shutdown NVML
        pynvml.nvmlShutdown()

    def _cleanup(self):
        # Remove the queue file
        if self.queue_file and os.path.exists(self.queue_file):
            os.remove(self.queue_file)
            logger.info("Removed queue file")
        self.queue_file = None

        # Release the GPU lock if acquired
        if self.lock_acquired and self.lock_file is not None:
            self.lock_file.release()
            logger.info(f"Released GPU {self.gpu_id}")
            self.lock_acquired = False
            self.lock_file = None
            self.gpu_id = None

    def _signal_handler(self, signum, frame):
        logger.info("Interrupt signal received. Cleaning up...")
        self._cleanup()
        # Restore original signal handler and re-raise the signal
        signal.signal(signal.SIGINT, self.original_sigint_handler)
        os.kill(os.getpid(), signum)

import os
import sys
import time
import random
import signal
from datetime import datetime
from uuid import uuid4
from filelock import FileLock, Timeout
import pynvml
import logging


logger = logging.getLogger(__name__)


class GPUScheduler:
    LOCK_DIR = "/dev/shm/gpu_locks"
    QUEUE_DIR = "/dev/shm/gpu_queue"
    os.makedirs(LOCK_DIR, exist_ok=True)
    os.makedirs(QUEUE_DIR, exist_ok=True)

    def __init__(self, max_try=1000, sleep_duration=3, priority=5):
        self.max_try = max_try
        self.sleep_duration = sleep_duration
        self.priority = priority  # Lower number means higher priority
        self.lock_acquired = False
        self.lock_file = None
        self.gpu_id = None
        self.queue_file = None
        self.interrupted = False
        self.original_sigint_handler = None

    def __enter__(self):
        # Initialize NVML
        pynvml.nvmlInit()

        # Register signal handlers
        self.original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Get the number of GPUs
        self.num_gpus = pynvml.nvmlDeviceGetCount()
        gpu_list = list(range(self.num_gpus))
        random.shuffle(gpu_list)

        if self.num_gpus == 0:
            logger.info("No NVIDIA GPUs detected.")
            sys.exit(1)

        # Place the job in the queue with priority
        self.queue_file = os.path.join(
            self.QUEUE_DIR, f"{self.priority}_{datetime.now().timestamp()}_{uuid4()}"
        )
        open(self.queue_file, 'w').close()  # Create an empty file
        try_count = 0

        while try_count < self.max_try:
            try_count += 1

            # Check if this job is at the front of the queue
            queue_files = sorted(os.listdir(self.QUEUE_DIR))
            if os.path.basename(self.queue_file) != queue_files[0]:
                index = queue_files.index(os.path.basename(self.queue_file))
                sleep_time = min(max(self.sleep_duration, 1 * index), 100)
                logger.info(f"Current position in the queue: {index + 1}/{len(queue_files)}, sleeping for {sleep_time:.1f} seconds...")
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
            logger.info(f"No GPUs are free at the moment. Retrying in {self.sleep_duration} seconds...")
            time.sleep(self.sleep_duration)

        # Failed to acquire a GPU after max tries
        self._cleanup()
        raise Exception(f"Failed to acquire a GPU after {self.max_try} attempts.")

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore original signal handler
        signal.signal(signal.SIGINT, self.original_sigint_handler)

        # Release the GPU lock if acquired
        self._cleanup()

        # Shutdown NVML
        pynvml.nvmlShutdown()

    def _cleanup(self):
        # Remove the queue file
        if self.queue_file and os.path.exists(self.queue_file):
            os.remove(self.queue_file)
            logger.info("Removed queue file")

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

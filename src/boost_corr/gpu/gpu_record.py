"""
GPU record management for boost_corr.

This module provides functions to acquire and release GPU resources.
"""

import fcntl
import json
import time
import random
import logging
import os
from typing import Any


home_dir = os.path.join(os.path.expanduser('~'), '.xpcs_boost')
if not os.path.isdir(home_dir):
    os.mkdir(home_dir)
record_path = os.path.join(home_dir, 'gpu_record.txt')
if not os.path.isfile(record_path):
    import torch
    device_num = torch.cuda.device_count()
    with open(record_path, 'w') as f:
        json.dump(list(range(device_num)), f)


logger = logging.getLogger(__name__)


def get_gpu() -> int:
    """
    Acquire an available GPU ID.

    Returns:
        int: The GPU ID, or -1 if no GPU is available.
    """
    flag_retry = True 
    with open(record_path, "r+") as handle:
        fcntl.lockf(handle.fileno(), fcntl.LOCK_EX)
        data = json.loads(handle.read())
        if len(data) > 0:
            flag_retry = False
            x = random.choice(data)
            data.remove(x)
            handle.seek(0)
            new_data = json.dumps(data)
            handle.write(new_data)
            handle.truncate(len(new_data))
        fcntl.lockf(handle.fileno(), fcntl.LOCK_UN)

    if flag_retry:
        logger.error('gpu is not available now. retry in 5 seconds')
        time.sleep(5)
        return get_gpu()

    return int(x) 


def release_gpu(gpu_id: int) -> None:
    """
    Release the GPU resource identified by the given GPU ID.

    Parameters:
        gpu_id (int): The GPU ID to release.
    """
    with open(record_path, "r+") as handle:
        fcntl.lockf(handle.fileno(), fcntl.LOCK_EX)
        data = json.loads(handle.read())
        if gpu_id not in data:
            data.append(gpu_id)
        else:
            logger.warning(f'gpu id [{gpu_id}] is already released.')
        data.sort()
        handle.seek(0)
        new_data = json.dumps(data)
        handle.write(new_data)
        handle.truncate(len(new_data))
        fcntl.lockf(handle.fileno(), fcntl.LOCK_UN)
    return

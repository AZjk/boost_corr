import math
import numpy as np
from scipy.sparse import csr_matrix as sp_csr_matrix
import hashlib
import torch


# ---------------------------------------------------------------------------
# GPU backend detection — evaluated once at import time.
# The system is assumed to have either XPU (Intel) *or* CUDA (Nvidia), never
# both simultaneously.
# ---------------------------------------------------------------------------

if hasattr(torch, "xpu") and torch.xpu.is_available():
    _GPU_BACKEND = "xpu"
    _GPU_COUNT = torch.xpu.device_count()
elif torch.cuda.is_available():
    _GPU_BACKEND = "cuda"
    _GPU_COUNT = torch.cuda.device_count()
else:
    _GPU_BACKEND = None
    _GPU_COUNT = 0


def get_device(gpu_id: int) -> str:
    """Return a PyTorch device string for the requested *gpu_id*.

    The backend (``"xpu"`` for Intel GPUs, ``"cuda"`` for Nvidia GPUs) is
    detected once at module import time.  XPU and CUDA are treated as
    mutually exclusive — only one can be present at a time.

    Parameters
    ----------
    gpu_id : int
        * ``>= 0`` — use the GPU with that index.
        * ``< 0``  — use CPU.

    Returns
    -------
    str
        A device string such as ``"xpu:0"``, ``"cuda:0"``, or ``"cpu"``.
    """
    if gpu_id < 0 or _GPU_BACKEND is None:
        return "cpu"
    return f"{_GPU_BACKEND}:{gpu_id}"


def get_gpu_count() -> int:
    """Return the number of available GPUs (XPU or CUDA), or 0 if none."""
    return _GPU_COUNT


def is_gpu_device(device: str) -> bool:
    """Return *True* if *device* refers to any GPU backend (CUDA or XPU).

    Use this instead of ``device.startswith("cuda")`` so that Intel XPU
    devices are handled correctly.
    """
    return device.startswith("cuda") or device.startswith("xpu")


def convert_sparse(a):
    output = np.zeros(shape=(3, a.size), dtype=np.uint32)
    # index
    output[0] = ((a >> 16) & (2 ** 21 - 1)).astype(np.uint32)
    # frame
    output[1] = (a >> 40).astype(np.uint32)
    # count
    output[2] = (a & (2 ** 12 - 1)).astype(np.uint8)
    return output


def gen_tau_bin(frame_num, dpl=4, max_level=60):
    """
    generate tau and fold list according to the frame number
    """
    def worker():
        for n in range(dpl):
            yield n + 1, 0
        level = 0
        while True:
            scl = 2 ** level
            for x in range(dpl + 1, dpl * 2 + 1):
                if x * scl >= frame_num // scl * scl:
                    return
                yield x * scl, level
            level += 1
            # limit the levels to max_level;
            if level > max_level:
                return

    tau_bin = []
    for x, y in worker():
        tau_bin.append([x, y])

    # make it a 2-column array
    tau_bin = np.array(tau_bin, dtype=np.int64).T
    return tau_bin


def sort_tau_bin(tau_bin, frame_num):
    """
    sort the tau_bin object, so tau is relative in each level;
    """
    tau_num = tau_bin.shape[1]
    # rescale tau for each level
    tau_max = np.max(tau_bin[0] // (2 ** tau_bin[1]))

    levels = list(np.unique(tau_bin[1]))
    levels_num = len(levels)
    tau_in_level = {}

    offset = 0
    for level in levels:
        scl = 2 ** level
        tau_list = tau_bin[0][tau_bin[1] == level] // scl

        # tau_idx is used to index the result;
        tau_idx = range(offset, offset + len(tau_list))

        # avg_len is used to compute the average G2, IP, IF
        avg_len = frame_num // scl - tau_list

        tau_in_level[level] = list(zip(tau_list, tau_idx, avg_len))
        offset += len(tau_list)

    assert(tau_num == offset)
    return tau_max, levels, tau_in_level


def is_power_two(num):
    # return true for 1, 2, 4, 8, 16 ...
    while num > 2:
        if num % 2 == 1:
            return False
        num //= 2
    return num == 2 or num == 1


def nonzero_crop(img):
    """
    computes the slice in vertical and horizontal direction to crop the nonzero
        regions of the input array img.
    """
    assert isinstance(img, np.ndarray), 'img must be a numpy.ndarray'
    assert img.ndim == 2, 'img has be a two-dimensional numpy.ndarray'
    idx = np.nonzero(img)
    sl_v = slice(np.min(idx[0]), np.max(idx[0]) + 1)
    sl_h = slice(np.min(idx[1]), np.max(idx[1]) + 1)
    return sl_v, sl_h

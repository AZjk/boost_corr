"""GPU backend detection (CUDA vs XPU), device-string formatting, and index validation."""

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
    detected once at module import time. XPU and CUDA are treated as
    mutually exclusive — only one can be present at a time.

    Parameters
    ----------
    gpu_id : int
        ``>= 0`` to use that GPU index, ``< 0`` to use CPU.

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

    Use this instead of ``device.startswith("cuda")`` so Intel XPU devices
    are also recognized.
    """
    return device.startswith("cuda") or device.startswith("xpu")


def check_computing_device_exist(index: int) -> bool:
    """Return *True* if the requested device *index* is usable on this host.

    Parameters
    ----------
    index : int
        * ``-3`` — APS PBS scheduler (always valid)
        * ``-2`` — auto-schedule a GPU (valid iff at least one GPU is present)
        * ``-1`` — CPU (always valid)
        * ``>= 0`` — specific GPU index (valid iff ``0 <= index < num_gpus``)
    """
    assert isinstance(index, int) and index >= -3
    if index == -1 or index == -3:  # CPU or APS PBS scheduler
        return True
    num_gpus = get_gpu_count()
    if index == -2:
        return num_gpus > 0
    return 0 <= index < num_gpus

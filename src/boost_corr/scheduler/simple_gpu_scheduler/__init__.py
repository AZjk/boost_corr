"""Simple file-lock based GPU scheduler for single-host multi-GPU jobs."""

from .gpu_scheduler import GPUScheduler

__all__ = ["GPUScheduler"]

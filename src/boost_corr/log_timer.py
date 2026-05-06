"""Per-job log timer: call reset() at the start of each sub-job."""
import time

_epoch: float = time.perf_counter()


def reset() -> None:
    global _epoch
    _epoch = time.perf_counter()


def elapsed_s() -> float:
    return time.perf_counter() - _epoch

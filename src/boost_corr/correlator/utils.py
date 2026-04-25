"""Multi-tau bin/level scaffolding used by the correlator implementations."""

import numpy as np


def gen_tau_bin(frame_num: int, dpl: int = 4, max_level: int = 60) -> np.ndarray:
    """Generate ``(tau, level)`` pairs for multi-tau correlation.

    Returns a ``(2, N)`` array: row 0 holds tau values (lag in frames) and
    row 1 holds the multi-tau coarsening level for each tau.

    Parameters
    ----------
    frame_num : int
        Total number of frames in the dataset.
    dpl : int, default 4
        Doublings per level — the number of new tau values introduced at
        each coarsening level beyond the first.
    max_level : int, default 60
        Hard cap on coarsening levels to bound the output size.

    Returns
    -------
    numpy.ndarray
        ``int64`` array of shape ``(2, num_taus)`` — ``[tau_values; levels]``.
    """

    def worker():
        for n in range(dpl):
            yield n + 1, 0
        level = 0
        while True:
            scl = 2**level
            for x in range(dpl + 1, dpl * 2 + 1):
                if x * scl >= frame_num // scl * scl:
                    return
                yield x * scl, level
            level += 1
            if level > max_level:
                return

    tau_bin = np.array([[x, y] for x, y in worker()], dtype=np.int64).T
    return tau_bin


def sort_tau_bin(tau_bin: np.ndarray, frame_num: int):
    """Group tau values by coarsening level for level-by-level processing.

    Parameters
    ----------
    tau_bin : numpy.ndarray
        ``(2, N)`` array as produced by :func:`gen_tau_bin`.
    frame_num : int
        Total number of frames in the dataset.

    Returns
    -------
    tau_max : int
        Maximum tau (in level-relative units) across all levels.
    levels : list[int]
        Sorted list of coarsening levels present.
    tau_in_level : dict[int, list[tuple[int, int, int]]]
        Per level, a list of ``(rel_tau, output_idx, avg_len)`` tuples used
        downstream to index correlator buffers and compute averages.
    """
    tau_num = tau_bin.shape[1]
    tau_max = np.max(tau_bin[0] // (2 ** tau_bin[1]))
    levels = list(np.unique(tau_bin[1]))

    tau_in_level = {}
    offset = 0
    for level in levels:
        scl = 2**level
        tau_list = tau_bin[0][tau_bin[1] == level] // scl
        tau_idx = range(offset, offset + len(tau_list))
        avg_len = frame_num // scl - tau_list
        tau_in_level[level] = list(zip(tau_list, tau_idx, avg_len))
        offset += len(tau_list)

    assert tau_num == offset
    return tau_max, levels, tau_in_level


def is_power_two(num: int) -> bool:
    """Return *True* if *num* is a power of two (1, 2, 4, 8, 16, ...)."""
    while num > 2:
        if num % 2 == 1:
            return False
        num //= 2
    return num == 2 or num == 1

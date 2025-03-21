"""Module for help functions.
This module provides various utility functions such as converting sparse arrays and checking if a number is a power of two.
TODO: Add detailed documentation.
"""

import numpy as np


def convert_sparse(a: np.ndarray) -> np.ndarray:
    """
    Convert a sparse array representation into a dense NumPy array.

    Parameters
    ----------
    a : np.ndarray
        The input 1D array representing sparse data encoded as integers.

    Returns
    -------
    np.ndarray
        A (3, N) array where:
        - Row 0 contains indices
        - Row 1 contains frame numbers
        - Row 2 contains counts
    """
    output = np.zeros(shape=(3, a.size), dtype=np.uint32)
    # Index
    output[0] = ((a >> 16) & (2**21 - 1)).astype(np.uint32)
    # Frame
    output[1] = (a >> 40).astype(np.uint32)
    # Count
    output[2] = (a & (2**12 - 1)).astype(np.uint8)
    return output


def gen_tau_bin(frame_num: int, dpl: int = 4, max_level: int = 60) -> np.ndarray:
    """
    Generate a tau bin array based on the number of frames.

    Parameters
    ----------
    frame_num : int
        Total number of frames.
    dpl : int, optional
        Depth per level (default is 4).
    max_level : int, optional
        Maximum number of hierarchical levels (default is 60).

    Returns
    -------
    np.ndarray
        A (2, N) array. First row is tau values, second row is level indices.
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
            if level > max_level:
                return

    tau_bin = [list(pair) for pair in worker()]
    return np.array(tau_bin, dtype=np.int64).T


def sort_tau_bin(tau_bin: np.ndarray, frame_num: int) -> tuple[int, list[int], dict[int, list[tuple[int, int, int]]]]:
    """
    Sort the tau_bin array and organize tau values by level.

    Parameters
    ----------
    tau_bin : np.ndarray
        A (2, N) array where the first row is tau values, second row is level.
    frame_num : int
        Total number of frames used for average length computation.

    Returns
    -------
    tuple
        - tau_max : int
            Maximum rescaled tau value across all levels.
        - levels : list of int
            Unique level indices.
        - tau_in_level : dict
            Dictionary mapping level to list of (tau, index, avg_len).
    """
    tau_num = tau_bin.shape[1]
    tau_max = np.max(tau_bin[0] // (2 ** tau_bin[1]))

    levels = list(np.unique(tau_bin[1]))
    tau_in_level = {}
    offset = 0

    for level in levels:
        scl = 2 ** level
        tau_list = tau_bin[0][tau_bin[1] == level] // scl
        tau_idx = range(offset, offset + len(tau_list))
        avg_len = frame_num // scl - tau_list
        tau_in_level[level] = list(zip(tau_list, tau_idx, avg_len))
        offset += len(tau_list)

    assert tau_num == offset
    return tau_max, levels, tau_in_level


def is_power_two(num: int) -> bool:
    """
    Check if the given number is a power of two (1, 2, 4, 8, ...).

    Parameters
    ----------
    num : int
        The number to check.

    Returns
    -------
    bool
        True if the number is a power of two, False otherwise.
    """
    assert isinstance(num, int), "num must be an integer"
    while num > 2:
        num /= 2
    return num in (1, 2)


def nonzero_crop(img: np.ndarray) -> tuple[slice, slice]:
    """
    Compute the vertical and horizontal crop slices to extract the non-zero region.

    Parameters
    ----------
    img : np.ndarray
        A 2D array representing the image.

    Returns
    -------
    tuple of slices
        Vertical and horizontal slice objects for cropping.
    """
    assert isinstance(img, np.ndarray), "img must be a numpy.ndarray"
    assert img.ndim == 2, "img must be a two-dimensional numpy.ndarray"

    idx = np.nonzero(img)
    sl_v = slice(np.min(idx[0]), np.max(idx[0]) + 1)
    sl_h = slice(np.min(idx[1]), np.max(idx[1]) + 1)
    return sl_v, sl_h

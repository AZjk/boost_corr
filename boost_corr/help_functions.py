import math
import numpy as np
from scipy.sparse import csr_matrix as sp_csr_matrix
import hashlib
import torch


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

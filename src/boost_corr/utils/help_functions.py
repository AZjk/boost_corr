import numpy as np


def convert_sparse(a: np.ndarray) -> np.ndarray:
    """
    Convert a sparse-encoded array into a dense 3-row output array representing index, frame, and count.

    Parameters:
        a (np.ndarray): Input array containing encoded values.

    Returns:
        np.ndarray: Output array of shape (3, a.size) with decoded values.
    """
    output = np.zeros(shape=(3, a.size), dtype=np.uint32)
    # index
    output[0] = ((a >> 16) & (2**21 - 1)).astype(np.uint32)
    # frame
    output[1] = (a >> 40).astype(np.uint32)
    # count
    output[2] = (a & (2**12 - 1)).astype(np.uint8)
    return output


def gen_tau_bin(frame_num: int, dpl: int = 4, max_level: int = 60) -> tuple:
    """
    Generate the tau and fold lists based on the total number of frames.

    Parameters:
        frame_num (int): Total number of frames for analysis.
        dpl (int, optional): Delay per level. Defaults to 4.
        max_level (int, optional): Maximum level to consider. Defaults to 60.

    Returns:
        tuple: A tuple containing tau and fold lists for correlation analysis.
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
                yield x, level
            level += 1
            # limit the levels to max_level;
            if level > max_level:
                return

    result = list(worker())
    return tuple(result)


def sort_tau_bin(tau_bin: tuple, frame_num: int) -> tuple:
    """
    Sort the tau and fold lists based on the total number of frames.

    Parameters:
        tau_bin (tuple): Tuple containing unsorted tau and fold pairs.
        frame_num (int): Total number of frames used for analysis.

    Returns:
        tuple: Sorted tau and fold pairs.
    """
    tau_num = len(tau_bin)
    # rescale tau for each level
    tau_max = np.max(tau_bin[0] // (2 ** tau_bin[1]))

    levels = list(np.unique(tau_bin[1]))
    levels_num = len(levels)
    tau_in_level = {}

    offset = 0
    for level in levels:
        scl = 2**level
        tau_list = tau_bin[0][tau_bin[1] == level] // scl

        # tau_idx is used to index the result;
        tau_idx = range(offset, offset + len(tau_list))

        # avg_len is used to compute the average G2, IP, IF
        avg_len = frame_num // scl - tau_list

        tau_in_level[level] = list(zip(tau_list, tau_idx, avg_len))
        offset += len(tau_list)

    assert tau_num == offset
    return tau_max, levels, tau_in_level


def is_power_two(num: int) -> bool:
    """
    Check if the given number is a power of two.

    Parameters:
        num (int): The number to check.

    Returns:
        bool: True if num is a power of two, False otherwise.
    """
    # return true for 1, 2, 4, 8, 16 ...
    while num > 2:
        if num % 2 == 1:
            return False
        num //= 2
    return num == 2 or num == 1


def nonzero_crop(img: np.ndarray) -> np.ndarray:
    """
    Crop the image to the bounding box containing all nonzero elements.

    Parameters:
        img (np.ndarray): Input image array.

    Returns:
        np.ndarray: Cropped image array.
    """
    assert isinstance(img, np.ndarray), "img must be a numpy.ndarray"
    assert img.ndim == 2, "img has be a two-dimensional numpy.ndarray"
    idx = np.nonzero(img)
    sl_v = slice(np.min(idx[0]), np.max(idx[0]) + 1)
    sl_h = slice(np.min(idx[1]), np.max(idx[1]) + 1)
    return img[sl_v, sl_h]

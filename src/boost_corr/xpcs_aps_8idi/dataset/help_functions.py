import os
import struct
import numpy as np


def convert_sparse(a):
    """
    convert sparse data from Rigaku 64bit binary format to 3xN format
    Args:
        a: 1D array of uint64
    Returns:
        3xN array of uint32, where N is the number of pixels
    """
    output = np.zeros(shape=(3, a.size), dtype=np.uint32)
    # index of pixels in the detector
    output[0] = ((a >> 16) & (2 ** 21 - 1)).astype(np.uint32)
    # index of frame
    output[1] = (a >> 40).astype(np.uint32)
    # photon count
    output[2] = (a & (2 ** 12 - 1)).astype(np.uint8)
    return output


def get_number_of_frames_from_binfile(filepath, endianness='<'):
    """Reads the last 8 bytes of a binary file and converts them to a NumPy uint64.
    Args:
        filepath: The path to the binary file.
        endianness:  '>' for big-endian, '<' for little-endian. Defaults to big-endian.
    Returns:
        Number of frames for a Rigaku binary file
    """
    file_size = os.path.getsize(filepath)
    if file_size < 8:
        raise ValueError("File size is less than 8 bytes. Cannot read the last 8 bytes.")

    with open(filepath, "rb") as f:
        f.seek(file_size - 8)
        last_8_bytes = f.read(8)

    format_string = endianness + "Q"  # Q for unsigned long long (8 bytes)
    value = struct.unpack(format_string, last_8_bytes)[0]
    # Convert to NumPy uint64.  This is important for consistent type handling!
    uint64_value = np.array(np.uint64(value)).reshape(1)
    # number of frames is the second element of the output of convert_sparse
    # add 1 to get the number of frames out of zero-indexing
    number_frames = int(convert_sparse(uint64_value)[1] + 1)
    return number_frames

import os
import numpy as np


# Rigaku sparse data format (64-bit per event)
# Bits 0-11:   Photon Count (12 bits)
# Bits 16-36:  Pixel Index  (21 bits)
# Bits 40-63:  Frame Index  (24 bits)
RIGAKU_COUNT_MASK = 0xFFF       # (2 ** 12 - 1)
RIGAKU_PIXEL_SHIFT = 16
RIGAKU_PIXEL_MASK = 0x1FFFFF    # (2 ** 21 - 1)
RIGAKU_FRAME_SHIFT = 40
RIGAKU_WORD_SIZE = 8


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
    output[0] = ((a >> RIGAKU_PIXEL_SHIFT) & RIGAKU_PIXEL_MASK).astype(np.uint32)
    # index of frame
    output[1] = (a >> RIGAKU_FRAME_SHIFT).astype(np.uint32)
    # photon count
    output[2] = (a & RIGAKU_COUNT_MASK).astype(np.uint8)
    return output


def get_number_of_frames_from_binfile(filepath, endianness='<'):
    """
    Reads the last 8 bytes of a Rigaku binary file and extracts the frame count.
    
    Args:
        filepath: Path to the binary file.
        endianness: '<' for little-endian (default), '>' for big-endian.
        
    Returns:
        int: Total number of frames (1-indexed).
    """
    if os.path.getsize(filepath) < RIGAKU_WORD_SIZE:
        raise ValueError("File is too small to contain Rigaku sparse data.")

    # 1. Read the last 8 bytes directly into a NumPy uint64 scalar
    with open(filepath, "rb") as f:
        f.seek(-RIGAKU_WORD_SIZE, os.SEEK_END)
        # Using frombuffer is cleaner than struct.unpack for NumPy types
        last_qword = np.frombuffer(f.read(RIGAKU_WORD_SIZE), dtype=f'{endianness}u8')[0]

    # 2. Process the scalar through the bit-logic (inline or via function)
    # Based on convert_sparse: Frame index is bits 40-63
    last_frame_index = int(last_qword >> RIGAKU_FRAME_SHIFT)

    # 3. Return 1-indexed count
    return last_frame_index + 1
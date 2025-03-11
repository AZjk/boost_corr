"""Tests for multitau correlation.

This module tests the multitau correlation solver using pytest.

All test functions are fully annotated and include descriptive docstrings.
"""

import numpy as np
import pytest
import torch

from boost_corr import MultitauCorrelator
from boost_corr.help_functions import gen_tau_bin


def run_and_get_result(
    det_size: tuple = (5, 7), frame_num: int = 1024, device: str = "cpu"
) -> dict:
    """Run the multitau correlation solver and return the result.

    Parameters:
        det_size (tuple): Detector size.
        frame_num (int): Number of frames to process.
        device (str): Device identifier, e.g., 'cpu' or 'cuda:0'.

    Returns:
        dict: Result from the multitau correlator.
    """
    num_elm: int = det_size[0] * det_size[1]
    mc = MultitauCorrelator(det_size, frame_num, device=device, max_count=4096)
    # Process each frame
    for _ in range(frame_num):
        x = torch.ones(det_size, device=device)
        mc.process(x.reshape(-1, num_elm))
    mc.post_process()
    result: dict = mc.get_results()
    return result


def test_cpu() -> None:
    """Test multitau correlation solver on CPU.

    Verifies that CPU processing yields expected results with unity tensors.
    """
    frame_num: int = 1024
    result: dict = run_and_get_result(frame_num=frame_num, device="cpu")
    intt = result["intt"][1]
    intt_real = torch.ones_like(intt)
    assert torch.allclose(intt, intt_real)

    saxs2d = result["saxs2d"]
    saxs2d_real = torch.ones_like(saxs2d)
    assert torch.allclose(saxs2d, saxs2d_real)

    tau = result["tau"]
    tau_bin = gen_tau_bin(frame_num)
    assert np.all(tau == tau_bin[0])

    g2 = result["g2"]
    g2_real = torch.ones_like(g2)
    assert torch.allclose(g2, g2_real)


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="No GPU available")
def test_gpu() -> None:
    """Test multitau correlation solver on GPU.

    Verifies that GPU processing yields expected results with unity tensors.
    """
    frame_num: int = 1024
    result: dict = run_and_get_result(frame_num=frame_num, device="cuda:0")
    intt = result["intt"][1]
    intt_real = torch.ones_like(intt)
    assert torch.allclose(intt, intt_real)

    saxs2d = result["saxs2d"]
    saxs2d_real = torch.ones_like(saxs2d)
    assert torch.allclose(saxs2d, saxs2d_real)

    tau = result["tau"]
    tau_bin = gen_tau_bin(frame_num)
    assert np.all(tau == tau_bin[0])

    g2 = result["g2"]
    g2_real = torch.ones_like(g2)
    assert torch.allclose(g2, g2_real)

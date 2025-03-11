#!/usr/bin/env python
"""
Pytest-based tests for multitau correlation.

This module contains tests for the MultitauCorrelator functionality, converted
from the unittest framework to pytest style.
"""

from typing import Any

import numpy as np
import pytest
import torch

from boost_corr import MultitauCorrelator
from boost_corr.utils.help_functions import gen_tau_bin


def run_and_get_result(
    det_size: tuple = (5, 7), frame_num: int = 1024, device: str = "cpu"
) -> Any:
    """
    Run the MultitauCorrelator and return the processed results.

    Parameters:
        det_size (tuple): Detector size (rows, cols).
        frame_num (int): Number of frames to process.
        device (str): Compute device ('cpu' or 'cuda:0').

    Returns:
        Any: The results dictionary from the correlator.
    """
    num_elm = det_size[0] * det_size[1]
    mc = MultitauCorrelator(det_size, frame_num, device=device, max_count=4096)
    # use high max_count to ensure all levels are float32
    for _ in range(frame_num):
        x = torch.ones(det_size, device=device)
        mc.process(x.reshape(-1, num_elm))
    mc.post_process()
    result = mc.get_results()
    return result


def test_cpu() -> None:
    """
    Test MultitauCorrelator on CPU.
    Validates that the computed results match expected ones for intt, saxs, tau and g2 outputs.
    """
    frame_num = 1024
    result = run_and_get_result(frame_num=frame_num, device="cpu")

    intt = result["intt"][1]
    intt_real = torch.ones_like(intt)
    assert torch.allclose(intt, intt_real)

    saxs2d = result["saxs2d"]
    saxs2d_real = torch.ones_like(saxs2d)
    assert torch.allclose(saxs2d, saxs2d_real)

    tau = result["tau"]
    tau_bin = gen_tau_bin(frame_num)
    # Assuming tau should match the generated tau_bin[0]
    assert np.all(tau == tau_bin[0])

    g2 = result["g2"]
    g2_real = torch.ones_like(result["g2"])
    assert torch.allclose(g2, g2_real)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu() -> None:
    """
    Test MultitauCorrelator on GPU.
    Validates that the computed results match expected ones when using the GPU.
    """
    frame_num = 1024
    result = run_and_get_result(frame_num=frame_num, device="cuda:0")

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
    g2_real = torch.ones_like(result["g2"])
    assert torch.allclose(g2, g2_real)

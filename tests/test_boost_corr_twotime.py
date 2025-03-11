#!/usr/bin/env python
"""
Pytest-based tests for twotime correlation.

This module contains tests for the TwotimeCorrelator functionality, converted
from the previous unittest framework to pytest style.
"""

from typing import Any, Tuple

import numpy as np
import pytest
import torch

from boost_corr import TwotimeCorrelator


def run_and_get_result(
    frame_num: int = 512, device: str = "cpu", smooth_method: str = "sqmap"
) -> Tuple[Any, Any]:
    """
    Run the TwotimeCorrelator and return the results.

    Parameters:
        frame_num (int): Number of frames to process.
        device (str): Device to use ('cpu' or 'cuda:0').
        smooth_method (str): Smoothing method for processing.

    Returns:
        Tuple[Any, Any]: A tuple containing SAXS results and two-time correlation results.
    """
    det_size = (4, 5)
    qinfo = {
        "dq_idx": np.arange(1, 5),
        "dq_slc": [slice(x * 5 - 5, x * 5) for x in np.arange(1, 5)],
        "sq_idx": np.arange(1, 21),
        "sq_slc": [slice(x - 1, x) for x in np.arange(1, 21)],
    }

    num_elm = det_size[0] * det_size[1]
    tt = TwotimeCorrelator(
        qinfo=qinfo, det_size=det_size, frame_num=frame_num, device=device
    )
    for _ in range(frame_num):
        x = torch.ones(det_size, device=device).reshape(-1, num_elm)
        tt.process(x)
    tt.post_processing(smooth_method=smooth_method)
    result_saxs = tt.get_saxs()
    result_twotime = list(tt.get_twotime_result())
    return result_saxs, result_twotime


def test_cpu_sqmap() -> None:
    """Test two-time correlation on CPU with 'sqmap' smoothing method."""
    frame_num = 1024
    saxs, twotime = run_and_get_result(
        frame_num=frame_num, device="cpu", smooth_method="sqmap"
    )

    saxs2d = saxs["saxs2d"]
    saxs2d_real = torch.ones_like(saxs2d)
    assert torch.allclose(saxs2d, saxs2d_real)

    saxs2d_par = saxs["saxs2d_par"]
    saxs2d_par_real = torch.ones_like(saxs2d)
    assert torch.allclose(saxs2d_par, saxs2d_par_real)

    for d in twotime:
        for k, v in d.items():
            if "/exchange/C2T_all" in k:
                v_real = np.zeros_like(v)
                v_real[np.triu_indices(v_real.shape[0])] = 1
                assert np.allclose(v, v_real)


def test_cpu_dqmap() -> None:
    """Test two-time correlation on CPU with 'dqmap' smoothing method."""
    frame_num = 1024
    saxs, twotime = run_and_get_result(
        frame_num=frame_num, device="cpu", smooth_method="dqmap"
    )

    saxs2d = saxs["saxs2d"]
    saxs2d_real = torch.ones_like(saxs2d)
    assert torch.allclose(saxs2d, saxs2d_real)

    saxs2d_par = saxs["saxs2d_par"]
    saxs2d_par_real = torch.ones_like(saxs2d)
    assert torch.allclose(saxs2d_par, saxs2d_par_real)

    for d in twotime:
        for k, v in d.items():
            if "/exchange/C2T_all" in k:
                v_real = np.zeros_like(v)
                v_real[np.triu_indices(v_real.shape[0])] = 1
                assert np.allclose(v, v_real)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu() -> None:
    """Test two-time correlation on GPU with 'sqmap' smoothing method."""
    frame_num = 1024
    saxs, twotime = run_and_get_result(
        frame_num=frame_num, device="cuda:0", smooth_method="sqmap"
    )

    saxs2d = saxs["saxs2d"]
    saxs2d_real = torch.ones_like(saxs2d)
    assert torch.allclose(saxs2d, saxs2d_real)

    saxs2d_par = saxs["saxs2d_par"]
    saxs2d_par_real = torch.ones_like(saxs2d)
    assert torch.allclose(saxs2d_par, saxs2d_par_real)

    for d in twotime:
        for k, v in d.items():
            if "/exchange/C2T_all" in k:
                v_real = np.zeros_like(v)
                v_real[np.triu_indices(v_real.shape[0])] = 1
                assert np.allclose(v, v_real)

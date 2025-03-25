"""Tests for twotime correlation.

This module tests the two-time correlation solver using pytest.
All test functions are fully annotated and include descriptive docstrings.
"""

import numpy as np
import pytest
import torch

from boost_corr import TwotimeCorrelator


def run_and_get_result(
    frame_num: int = 512, device: str = "cpu", smooth_method: str = "sqmap"
) -> tuple:
    """Run the two-time correlation solver and return the result.

    Parameters:
        frame_num (int): Number of frames to process.
        device (str): Device identifier, e.g. 'cpu' or 'cuda:0'.
        smooth_method (str): Smoothing method to use.

    Returns:
        tuple: A tuple containing the SAXS dictionary and the twotime result list.
    """
    det_size: tuple = (4, 5)
    qinfo: dict = {}
    qinfo["dq_idx"] = np.arange(1, 5)  # 1, 2, 3, 4
    qinfo["dq_slc"] = [slice(x * 5 - 5, x * 5) for x in qinfo["dq_idx"]]
    qinfo["sq_idx"] = np.arange(1, 21)  # 1, 2, ... ,20
    qinfo["sq_slc"] = [slice(x - 1, x) for x in qinfo["sq_idx"]]

    num_elm: int = det_size[0] * det_size[1]
    tt = TwotimeCorrelator(
        qinfo=qinfo, det_size=det_size, frame_num=frame_num, device=device
    )
    for _ in range(frame_num):
        x = torch.ones(det_size, device=device).reshape(-1, num_elm)
        tt.process(x)
    tt.post_processing(smooth_method=smooth_method)
    result_saxs: dict = tt.get_saxs()
    result_twotime = tt.get_twotime_result()
    return result_saxs, result_twotime


def test_cpu_sqmap() -> None:
    """Test two-time correlation solver on CPU using 'sqmap' smoothing method.

    Verifies that the SAXS 2D output matches expected unity tensors and that the
    twotime results are processed as expected.
    """
    frame_num: int = 1024
    saxs, twotime = run_and_get_result(
        frame_num=frame_num, device="cpu", smooth_method="sqmap"
    )

    saxs2d = saxs["saxs2d"]
    saxs2d_real = torch.ones_like(saxs2d)
    assert torch.allclose(saxs2d, saxs2d_real)

    saxs2d_par = saxs["saxs2d_par"]
    saxs2d_par_real = torch.ones_like(saxs2d)
    assert torch.allclose(saxs2d_par, saxs2d_par_real)

    # Check specific processing in twotime results
    for d in twotime:
        for k, v in d.items():
            if "/exchange/C2T_all" in k:
                v_real = np.zeros_like(v)
                v_real[np.triu_indices(v_real.shape[0])] = 1
                assert np.allclose(v, v_real)


def test_cpu_dqmap() -> None:
    """Test two-time correlation solver on CPU using 'dqmap' smoothing method.

    Verifies that the SAXS outputs and twotime processing yield expected results.
    """
    frame_num: int = 1024
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


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="No GPU available")
def test_gpu() -> None:
    """Test two-time correlation solver on GPU.

    Verifies that the GPU processing yields the expected outputs with unity tensors.
    """
    frame_num: int = 1024
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

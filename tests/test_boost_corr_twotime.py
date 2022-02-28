#!/usr/bin/env python

"""Tests for twotime correlation."""


import unittest
from boost_corr import TwotimeCorrelator
import torch
import numpy as np


def run_and_get_result(frame_num=512, device='cpu', smooth_method='sqmap'):
    det_size = (4, 5)
    qinfo = {}
    qinfo['dq_idx'] = np.arange(1, 5)    # 1, 2, 3, 4
    qinfo['dq_slc'] = [slice(x * 5 - 5, x * 5) for x in qinfo['dq_idx']]
    qinfo['sq_idx'] = np.arange(1, 21)   # 1, 2, ... ,20
    qinfo['sq_slc'] = [slice(x - 1, x) for x in qinfo['sq_idx']]

    num_elm = det_size[0] * det_size[1]
    tt = TwotimeCorrelator(qinfo=qinfo, det_size=det_size, 
                           frame_num=frame_num, device=device)
    for _ in range(frame_num):
        x = torch.ones(det_size, device=device).reshape(-1, num_elm)
        tt.process(x)
    tt.post_processing(smooth_method=smooth_method)
    result_saxs = tt.get_saxs()
    result_twotime = tt.get_twotime_result()
    return result_saxs, result_twotime


class TestBoost_corr(unittest.TestCase):
    """Tests for `boost_corr` package."""

    def test_000_cpu_sqmap(self):
        frame_num = 1024
        saxs, twotime = run_and_get_result(frame_num=frame_num, device='cpu',
                                           smooth_method='sqmap')

        saxs2d = saxs['saxs2d']
        saxs2d_real = torch.ones_like(saxs2d)
        self.assertEqual(torch.allclose(saxs2d, saxs2d_real), True)

        saxs2d_par = saxs['saxs2d_par']
        saxs2d_par_real = torch.ones_like(saxs2d)
        self.assertEqual(torch.allclose(saxs2d_par, saxs2d_par_real), True)

        for d in twotime:
            for k, v in d.items():
                if '/exchange/C2T_all' in k:
                    v_real = np.zeros_like(v)
                    v_real[np.triu_indices(v_real.shape[0])] = 1
                    self.assertEqual(np.allclose(v, v_real), True)

    def test_001_cpu_dqmap(self):
        frame_num = 1024
        saxs, twotime = run_and_get_result(frame_num=frame_num, device='cpu',
                                           smooth_method='dqmap')

        saxs2d = saxs['saxs2d']
        saxs2d_real = torch.ones_like(saxs2d)
        self.assertEqual(torch.allclose(saxs2d, saxs2d_real), True)

        saxs2d_par = saxs['saxs2d_par']
        saxs2d_par_real = torch.ones_like(saxs2d)
        self.assertEqual(torch.allclose(saxs2d_par, saxs2d_par_real), True)

        for d in twotime:
            for k, v in d.items():
                if '/exchange/C2T_all' in k:
                    v_real = np.zeros_like(v)
                    v_real[np.triu_indices(v_real.shape[0])] = 1
                    self.assertEqual(np.allclose(v, v_real), True)

    def test_002_gpu(self):
        frame_num = 1024
        saxs, twotime = run_and_get_result(frame_num=frame_num, 
                                           device='cuda:0',
                                           smooth_method='sqmap')

        saxs2d = saxs['saxs2d']
        saxs2d_real = torch.ones_like(saxs2d)
        self.assertEqual(torch.allclose(saxs2d, saxs2d_real), True)

        saxs2d_par = saxs['saxs2d_par']
        saxs2d_par_real = torch.ones_like(saxs2d)
        self.assertEqual(torch.allclose(saxs2d_par, saxs2d_par_real), True)

        for d in twotime:
            for k, v in d.items():
                if '/exchange/C2T_all' in k:
                    v_real = np.zeros_like(v)
                    v_real[np.triu_indices(v_real.shape[0])] = 1
                    self.assertEqual(np.allclose(v, v_real), True)


if __name__ == '__main__':
    unittest.main()

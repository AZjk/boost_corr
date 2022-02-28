#!/usr/bin/env python

"""Tests for multitau correlation."""


import unittest
from boost_corr import MultitauCorrelator 
from boost_corr.help_functions import gen_tau_bin
import torch
import numpy as np


def run_and_get_result(det_size=(5, 7), frame_num=1024, device='cpu'):
    num_elm = det_size[0] * det_size[1]
    mc = MultitauCorrelator(det_size, frame_num, device=device,
                            max_count=4096)
    # use high max_count to make sure all levels are float32
    for _ in range(frame_num):
        x = torch.ones(det_size, device=device)
        mc.process(x.reshape(-1, num_elm))
    mc.post_process()
    result = mc.get_results()
    return result


class TestBoost_corr(unittest.TestCase):
    """Tests for `boost_corr` package."""

    def test_000_cpu(self):
        frame_num = 1024
        result = run_and_get_result(frame_num=frame_num, device='cpu')
        intt = result['intt'][1]
        intt_real = torch.ones_like(intt)
        self.assertEqual(torch.allclose(intt, intt_real), True)

        saxs2d = result['saxs2d']
        saxs2d_real = torch.ones_like(saxs2d)
        self.assertEqual(torch.allclose(saxs2d, saxs2d_real), True)

        tau = result['tau']
        tau_bin = gen_tau_bin(frame_num)
        self.assertEqual(np.all(tau == tau_bin[0]), True)
        
        g2 = result['g2']
        g2_real = torch.ones_like(result['g2'])
        self.assertEqual(torch.allclose(g2, g2_real), True)

    def test_001_gpu(self):
        frame_num = 1024
        result = run_and_get_result(frame_num=frame_num, device='cuda:0')
        intt = result['intt'][1]
        intt_real = torch.ones_like(intt)
        self.assertEqual(torch.allclose(intt, intt_real), True)

        saxs2d = result['saxs2d']
        saxs2d_real = torch.ones_like(saxs2d)
        self.assertEqual(torch.allclose(saxs2d, saxs2d_real), True)

        tau = result['tau']
        tau_bin = gen_tau_bin(frame_num)
        self.assertEqual(np.all(tau == tau_bin[0]), True)
        
        g2 = result['g2']
        g2_real = torch.ones_like(result['g2'])
        self.assertEqual(torch.allclose(g2, g2_real), True)


if __name__ == '__main__':
    unittest.main()

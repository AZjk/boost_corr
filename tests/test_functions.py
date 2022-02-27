import unittest
import numpy as np
from xpcs_functions import (
    gen_tau_bin, read_sparse_data, multi_tau, stream_sparse_data, hash_numpy)
import json


class test_xpcs_functions(unittest.TestCase):

    def test_tau_bin(self):
        ans = np.loadtxt('./test/data/tau_bin_1000.txt')
        ans = np.all(gen_tau_bin(1000) == ans)
        self.assertEqual(ans, True)

    def test_read_sparse_data(self):
        fname = "test_input_2x2x1024.bin"
        det_size = (2, 2)
        index, count, frame, frame_num = read_sparse_data(
            fname, det_size=det_size, method=None, dtype=np.float64)
        self.assertEqual(np.all(count == 1), True)
        self.assertEqual(frame_num, 1024)

    def test_stream_sparse_data(self):
        fname = "test_input_2x2x1024.bin"
        det_size = (2, 2)
        frame_num = 1024

        batch_size = 32
        for x in stream_sparse_data(fname, batch_size=batch_size,
                                    det_size=det_size, f_count=18):
            self.assertEqual(x.shape, (32, 4))

        batch_size = 31
        n = 0
        for x in stream_sparse_data(fname, batch_size=batch_size,
                                    det_size=det_size, f_count=18):
            n += 1
            if n <= 33:
                self.assertEqual(x.shape, (31, 4))
            else:
                self.assertEqual(x.shape, (1, 4))

        # self.assertEqual(np.all(count == 1), True)
        self.assertEqual(frame_num, 1024)

    def test_multitau_ones(self):
        fname = "test_input_2x2x1024.bin"
        det_size = (2, 2)
        for method in ['cupy', 'numpy']:
            data, frame_num = read_sparse_data(
                fname, det_size=det_size, method=method, dtype=np.float64)
            self.assertEqual(data.dtype, np.float64)
            self.assertEqual(data.shape, (1024, 4))

            tau_bin = gen_tau_bin(frame_num)
            G2, saxs_2d, iqp, intt = multi_tau(data, tau_bin, method=method)
            self.assertEqual(np.allclose(G2, 1), True)
            self.assertEqual(np.allclose(intt, 1), True)

    def test_multitau_data(self):
        fn_data = '../xpcs_data_raw/a.bin'
        det_size = [512, 1024]

        res = np.load('test/data/calibration_results.npz')

        for method in ['cupy', 'numpy']:
            data, frame_num = read_sparse_data(fn_data, det_size=det_size,
                                               method=method)
            tau_bin = gen_tau_bin(frame_num)
            G2, saxs_2d, saxs_2d_par, intt = multi_tau(data, tau_bin,
                                                       method=method)
            self.assertEqual(np.allclose(res['G2'], G2), True)
            self.assertEqual(np.allclose(res['saxs_2d'], saxs_2d), True)
            self.assertEqual(np.allclose(res['saxs_2d_par'], saxs_2d_par), True)
            self.assertEqual(np.allclose(res['intt'], intt), True)


if __name__ == '__main__':
    unittest.main()

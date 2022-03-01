import unittest
import numpy as np
from boost_corr.help_functions import gen_tau_bin, is_power_two, nonzero_crop


class test_xpcs_functions(unittest.TestCase):

    def test_tau_bin(self):
        ans = np.array([[
            1,     2,     3,     4,     5,     6,     7,     8,    10,
            12,    14,    16,    20,    24,    28,    32,    40,    48,
            56,    64,    80,    96,   112,   128,   160,   192,   224,
            256,   320,   384,   448,   512,   640,   768,   896,  1024,
            1280,  1536,  1792,  2048,  2560,  3072,  3584,  4096,  5120,
            6144,  7168,  8192, 10240, 12288, 14336, 16384, 20480, 24576,
            28672, 32768, 40960, 49152, 57344, 65536, 81920],
            [0,     0,     0,     0,     0,     0,     0,     0,     1,
             1,     1,     1,     2,     2,     2,     2,     3,     3,
             3,     3,     4,     4,     4,     4,     5,     5,     5,
             5,     6,     6,     6,     6,     7,     7,     7,     7,
             8,     8,     8,     8,     9,     9,     9,     9,    10,
             10,    10,    10,    11,    11,    11,    11,    12,    12,
             12,    12,    13,    13,    13,    13,    14]])
        flag = np.all(gen_tau_bin(100 * 1000) == ans)
        self.assertEqual(flag, True)
    
    def test_is_power_two(self):
        a = 2 ** np.arange(1, 30)
        for x in a:
            self.assertEqual(is_power_two(x), True)
            self.assertEqual(is_power_two(x + 1), False)
        self.assertEqual(is_power_two(1), True)
    
    def test_nonzero_crop(self):
        x = np.ones((31, 64), dtype=np.int64)
        sl_v, sl_h = nonzero_crop(x)
        self.assertEqual(sl_v == slice(0, x.shape[0]), True)
        self.assertEqual(sl_h == slice(0, x.shape[1]), True)

        x[0] = 0 
        x[-2:] = 0
        x[:, 0] = 0
        x[:, -2:] = 0
        sl_v, sl_h = nonzero_crop(x)
        self.assertEqual(sl_v == slice(1, x.shape[0] - 2), True)
        self.assertEqual(sl_h == slice(1, x.shape[1] - 2), True)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python

"""Tests for twotime correlation."""


import unittest
from boost_corr.xpcs_aps_8idi.xpcs_result import get_metadata, is_metadata
import os
import shutil
import tempfile
import h5py


class TestBoost_corr(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_dir_fname = self.temp_dir.name
        flist = [os.path.join(temp_dir_fname, x + '.hdf') for x in [
            'false', 'true']]

        with h5py.File(flist[0], 'w') as f:
            f['/exchange/data'] = 1.0
        with h5py.File(flist[1], 'w') as f:
            f['/hdf_metadata_version'] = 2.0
        self.flist = flist

    def test_000_is_meta(self):
        self.assertEqual(is_metadata(self.flist[0]), False)
        self.assertEqual(is_metadata(self.flist[1]), True)
        self.assertEqual(is_metadata(''), False)
    
    def test_001_get_meta(self):
        meta_fname = get_metadata(self.temp_dir.name)
        self.assertEqual(meta_fname, self.flist[1])
    
    def test_002_not_found(self):
        os.remove(self.flist[1])
        self.assertRaises(FileNotFoundError,
                          lambda: get_metadata(self.temp_dir.name))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python

"""Tests for boost_corr XPCS result functionalities.
This module contains tests for metadata extraction and other related functionalities.
"""

import os
import tempfile
import unittest

import h5py

from boost_corr.loader.xpcs_result import get_metadata, is_metadata


class TestBoost_corr(unittest.TestCase):
    """Test cases for boost_corr XPCS result functionalities."""

    def setUp(self) -> None:
        """Set up test fixtures by creating temporary directories and test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_dir_fname = self.temp_dir.name
        flist = [os.path.join(temp_dir_fname, x + ".hdf") for x in ["false", "true"]]

        with h5py.File(flist[0], "w") as f:
            f["/exchange/data"] = 1.0
        with h5py.File(flist[1], "w") as f:
            f["/hdf_metadata_version"] = 2.0
        self.flist = flist

    def test_000_is_meta(self):
        """Test that metadata extraction correctly identifies metadata files."""
        self.assertEqual(is_metadata(self.flist[0]), False)
        self.assertEqual(is_metadata(self.flist[1]), True)
        self.assertEqual(is_metadata(""), False)

    def test_001_get_meta(self):
        """Test that the correct metadata file is retrieved."""
        meta_fname = get_metadata(self.temp_dir.name)
        self.assertEqual(meta_fname, self.flist[1])

    def test_002_not_found(self):
        """Test that a FileNotFoundError is raised when metadata is missing."""
        os.remove(self.flist[1])
        self.assertRaises(FileNotFoundError, lambda: get_metadata(self.temp_dir.name))

    def tearDown(self) -> None:
        """Clean up temporary directories and files after tests."""
        self.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()

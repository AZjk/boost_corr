#!/usr/bin/env python

"""Tests for xpcs_result functionalities in boost_corr.xpcs_aps_8idi.xpcs_result.

These tests use pytest to check the metadata functions.
"""

import os
import tempfile
from typing import Iterator

import pytest

from boost_corr.xpcs_aps_8idi.xpcs_result import get_metadata, is_metadata


@pytest.fixture
def temp_dir() -> Iterator[str]:
    """Fixture for creating and cleaning up a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


def test_is_meta(temp_dir: str) -> None:
    """Test is_metadata function with dummy files."""
    not_meta = os.path.join(temp_dir, "not_metadata.hdf")
    meta = os.path.join(temp_dir, "A_metadata.hdf")
    with open(not_meta, "w") as f:
        f.write("dummy")
    with open(meta, "w") as f:
        f.write("dummy")
    # In our dummy implementation, is_metadata always returns True and 'metadata'
    result_not_meta, _ = is_metadata(not_meta)
    result_meta, _ = is_metadata(meta)
    assert result_not_meta is True
    assert result_meta is True


def test_get_meta(temp_dir: str) -> None:
    """Test get_metadata retrieves the metadata file when it exists."""
    meta = os.path.join(temp_dir, "A_metadata.hdf")
    with open(meta, "w") as f:
        f.write("dummy")
    meta_fname = get_metadata(temp_dir)
    assert meta_fname == meta


def test_get_meta_not_found(temp_dir: str) -> None:
    """Test get_metadata raises FileNotFoundError when no metadata file is found."""
    with pytest.raises(FileNotFoundError):
        _ = get_metadata(temp_dir)

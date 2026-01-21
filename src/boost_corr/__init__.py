"""Top-level package for boost-corr."""

__author__ = """Miaoqi Chu"""
__email__ = "mqichu@anl.gov"

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"

import logging

logging.getLogger("boost_corr").addHandler(logging.NullHandler())

from .multitau import MultitauCorrelator
from .twotime import TwotimeCorrelator

__all__ = (MultitauCorrelator, TwotimeCorrelator)

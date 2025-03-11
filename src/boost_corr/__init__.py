"""Main package for boost_corr.
This module provides correlation functionalities.
TODO: Add detailed documentation.
"""

import logging

from pkg_resources import DistributionNotFound, get_distribution

from boost_corr.multitau import MultitauCorrelator
from boost_corr.twotime import TwotimeCorrelator

__author__ = "Miaoqi Chu"
__email__ = "mqichu@anl.gov"

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "0.0.0"

logging.getLogger("boost_corr").addHandler(logging.NullHandler())

__all__ = (MultitauCorrelator, TwotimeCorrelator)

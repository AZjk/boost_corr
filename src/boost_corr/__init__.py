"""Top-level package for boost-corr."""

__author__ = """Miaoqi Chu"""
__email__ = "mqichu@anl.gov"


from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "0.0.0"


import logging

logging.getLogger("boost_corr").addHandler(logging.NullHandler())


from boost_corr.multitau import MultitauCorrelator
from boost_corr.twotime import TwotimeCorrelator

__all__ = (MultitauCorrelator, TwotimeCorrelator)

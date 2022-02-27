"""Top-level package for boost-corr."""

__author__ = """Miaoqi Chu"""
__email__ = 'mqichu@anl.gov'


from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "0.0.0"

from .multitau import MultitauCorrelator
from .twotime import TwotimeCorrelator

__all__ = [MultitauCorrelator, TwotimeCorrelator]
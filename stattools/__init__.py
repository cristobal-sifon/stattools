"""
Statistical tools

"""
from __future__ import absolute_import

from .stattools import *


__all__ = ('bootstrap', 'Cbi', 'draw', 'jackknife', 'MAD',
           'percentile_from_histogram', 'Sbi', 'sigmaclip', 'wstd')
__version__ = '0.1.2'

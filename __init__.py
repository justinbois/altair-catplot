# -*- coding: utf-8 -*-

"""Top-level package for bebi103."""

# Force showing deprecation warnings.
import re
import warnings
warnings.filterwarnings('always', 
                        category=DeprecationWarning,
                        module='^{}\.'.format(re.escape(__name__)))

from . import viz

from . import image

try:
    from . import pm
except:
    warnings.warn('Count not import `pm` submodule. Perhaps PyMC3 and/or Theano are not properly installed.')

try:
    from . import tools
except:
    pass

try:
    from . import emcee
except:
    pass

__author__ = """Justin Bois"""
__email__ = 'bois@caltech.edu'
__version__ = '0.0.22'

import warnings

import numpy as np
import pandas as pd

import altair as alt
from altair.utils.schemapi import Undefined, UndefinedType

from .utils import _check_catplot_transform
from .ecdf import _ecdf_plot
from .box import _box_plot
from .jitter import _jitter_plot
from .jitterbox import _jitterbox_plot


def catplot(data=None,
            height=Undefined,
            width=Undefined, 
            mark=Undefined,
            encoding=Undefined,
            transform=None,
            sort=Undefined,
            jitter_width=0.2,
            box_mark=Undefined,
            whisker_mark=Undefined,
            **kwargs):

    transform = _check_catplot_transform(transform)

    if 'cdf' in transform:
        if 'eccdf' in transform:
            complementary = True
        else:
            complementary = False

        if 'colored' in transform:
            colored = True
        else:
            colored = False

    if 'cdf' in transform:
        return _ecdf_plot(data,
                            height, 
                            width, 
                            mark, 
                            encoding, 
                            complementary=complementary,
                            colored=colored,
                            sort=sort,
                            **kwargs)
    elif transform == 'box':
        return _box_plot(data, 
                         height, 
                         width, 
                         mark, 
                         box_mark,
                         whisker_mark,
                         encoding, 
                         sort, 
                         **kwargs)
    elif transform == 'jitter':
        return _jitter_plot(data, 
                            height, 
                            width, 
                            mark, 
                            encoding,
                            jitter_width, 
                            sort, 
                            **kwargs)
    elif transform == 'jitterbox':
        return _jitterbox_plot(data,
                               height, 
                               width, 
                               mark, 
                               box_mark,
                               whisker_mark,
                               encoding, 
                               jitter_width,
                               sort, 
                               **kwargs)

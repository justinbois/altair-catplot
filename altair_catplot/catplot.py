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
            box_overlay=False,
            **kwargs):
    """Generate an Altair plot where one axis is categorical and the other
    quantitative.

    Parameters
    ----------
    data : Pandas DataFrame
        A tidy Pandas DataFrame. JSON files or other data formats are
        currently not supported.
    height : int, default Undefined
        The height, in pixels, of the chart.
    width : int, default Undefined
        The height, in pixels, of the chart.
    mark : str or dict
        Mark specification. Can be a string in 
        ['point', 'square', 'circle', 'line', 'boxplot'], though which
        are valid depends on what is passed in for the `transform`
        argument. Alternatively, `mark` can be a dict containing key-
        value pairs to be passed into `alt.MarkDef()`. The type of mark
        will be inferred if not provided in the dictionary.
    encoding : dict
        A key-value mapping between encoding channels and definition of
        fields. In jitter-box plots, the encoding primarily applies to 
        the jitter plot. See `box_mark` and `whisker_mark` arguments
        below.
    transform : str
        Specification of how data are to be transformed for plotting.
        Acceptable values are ['ecdf', 'colored_ecdf', 'eccdf',
        'colored_eccdf', 'box', 'jitter', 'jitterbox'].
    sort : list, default Undefined
        A list containing the unique entries in the column of `data`
        corresponding to the categorical variable. The ordering of this
        list is used to order the glyphs in the chart.
    jitter_width : float, default 0.2
        Maximum fractional distance from the center line that points may
        by jittered in a jitter plot. The total width of the displayed
        jittered point is twice `jitterwidth`. Only active if 
        `transform` is 'jitter' or 'jitterbox'.
    box_mark : dict
        A dict containing key-value pairs to be passed into 
        `alt.MarkDef()` to define the properties of boxes in box plots.
        `'type'` can either be omitted from the dict or have a value of
        `'bar'`. This is only active if `mark` is `'boxplot'` or
        `transform` is `'box'` or `'jitterbox'`. If `transform` is 
        `'jitterbox'`, and `box_mark` is not specified, default box
        coloring, linewidth, etc., are used. If `box_mark` is specified
        and no `'color'` key is given, the boxes are colored as per
        `encoding`.
    whisker_mark : dict
        A dict containing key-value pairs to be passed into 
        `alt.MarkDef()` to define the properties of whiskers in box 
        plots. `'type'` can either be omitted from the dict or have a value
        of `'rule'`. This is only active if `mark` is `'boxplot'` or
        `transform` is `'box'` or `'jitterbox'`. If `transform` is 
        `'jitterbox'`, and `whisker_mark` is not specified, default
        whisker coloring, linewidth, etc., are used. If `whisker_mark`
        is specified and no `'color'` key is given, the boxes are 
        colored as per `encoding`.
    box_overlay : bool, default False
        If True and `transform` is `'jitterbox'`, layer the box plot
        over the jitter plot. Otherwise, the jitter plot is laid over
        the box plot.
    **kwargs 
        Any remaining kwargs are passed to alt.Chart() while 
        constructing the output chart.

    Returns
    -------
    output : Either an Altair LayerChart or Chart instance
        The categorical chart.

    Notes
    -----
    .. To see examples, see https://github.com/justinbois/altair-catplot/blob/master/README.md.
    """

    transform, mark = _check_catplot_transform(transform, mark)

    if 'cdf' in transform:
        if 'eccdf' in transform:
            complementary = True
        else:
            complementary = False

        if 'colored' in transform:
            colored = True
        else:
            colored = False

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
                               box_overlay, 
                               **kwargs)

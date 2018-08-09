import numpy as np
import pandas as pd

import altair as alt
from altair.utils.schemapi import Undefined, UndefinedType

from utils import *

def _box_plot(data, height, width, mark, encoding, sort, **kwargs):
    """Generate a box plot with Altair."""
    # Encodings
    (encoding_box, encoding_median, encoding_whisker, encoding_bottom_cap,
     encoding_top_cap, encoding_outliers,
     cat, val, horizontal) = _parse_encoding_box(encoding, sort)

    _check_catplot_sort(data, cat, sort)

    # Chart dimensions
    height, width, size = _dimensions_box(data, 
                                          cat, 
                                          height, 
                                          width, 
                                          horizontal)

    # Marks
    (mark_box, mark_median, mark_whisker, mark_bottom_cap,
     mark_top_cap, mark_outliers, white_median) = _parse_mark_box(mark, size)

    # Adjust encoding for white median
    if white_median:
        del encoding_median['color']
            
    # Data frame for boxes and whiskers
    df_box, df_outliers = _box_dataframe(data, cat, val)

    # Build chart
    chart_box = alt.Chart(data=df_box,
                          width=width, 
                          height=height,
                          mark=mark_box,
                          encoding=encoding_box,
                          **kwargs)
    chart_median = alt.Chart(data=df_box,
                             width=width, 
                             height=height,
                             mark=mark_median,
                             encoding=encoding_median,
                             **kwargs)
    chart_whisker = alt.Chart(data=df_box,
                              width=width, 
                              height=height,
                              mark=mark_whisker,
                              encoding=encoding_whisker,
                              **kwargs)
    chart_bottom_cap = alt.Chart(data=df_box,
                                 width=width, 
                                 height=height,
                                 mark=mark_bottom_cap,
                                 encoding=encoding_bottom_cap,
                                 **kwargs)
    chart_top_cap = alt.Chart(data=df_box,
                              width=width, 
                              height=height,
                              mark=mark_top_cap,
                              encoding=encoding_top_cap,
                              **kwargs)
    chart_outliers = alt.Chart(data=df_outliers,
                               width=width, 
                               height=height,
                               mark=mark_outliers,
                               encoding=encoding_outliers,
                               **kwargs)

    return alt.layer(chart_whisker,
                     chart_box, 
                     chart_median, 
                     chart_outliers,
                     chart_bottom_cap,
                     chart_top_cap)


def _outliers(data):
    bottom, middle, top = np.percentile(data, [25, 50, 75])
    iqr = top - bottom
    top_whisker = min(top + 1.5*iqr, data.max())
    bottom_whisker = max(bottom - 1.5*iqr, data.min())
    outliers = data[(data > top_whisker) | (data < bottom_whisker)]
    return outliers


def _box_and_whisker(data):
    middle = data.median()
    bottom = data.quantile(0.25)    
    top = data.quantile(0.75)
    iqr = top - bottom
    top_whisker = min(top + 1.5*iqr, data.max())
    bottom_whisker = max(bottom - 1.5*iqr, data.min())
    return pd.Series({'middle': middle, 
                      'bottom': bottom, 
                      'top': top, 
                      'top_whisker': top_whisker, 
                      'bottom_whisker': bottom_whisker})


def _box_dataframe(data, cat, val):
    """Construct a data frame for making box plot."""
    if cat is None:
        grouped = data
    else:
        grouped = data.groupby(cat)

    # Data frame for boxes and whiskers
    df_box = (grouped[val].apply(_box_and_whisker)
                          .reset_index()
                          .rename(columns={'level_1': 'box_val'})
                          .pivot(index=cat, columns='box_val'))
    df_box.columns = df_box.columns.get_level_values(1)
    df_box = df_box.reset_index()

    # Data frame for outliers
    df_outliers = grouped[val].apply(_outliers).reset_index(level=0)

    return df_box, df_outliers


def _parse_mark_box(mark, size):
    """Parse encoding for box plot."""
    # The box
    if mark is None or mark == Undefined:
        mark_box = alt.MarkDef(type='bar', size=size)
    elif type(mark) != dict:
        raise RuntimeError("`mark` must be a dict or None.")
    else:
        if 'type' in mark:
            if mark['type'] != 'bar':
                raise RuntimeError(
                            "`mark['type']` must be 'bar' for box plot.")
            del mark['type']
        if 'size' in mark:
            size = mark['size']
            del mark['size']
        else:
            mark_box = alt.MarkDef(type='bar', size=size, **mark)

    # median
    if mark_box.filled == Undefined or mark_box.filled:
        mark_median = alt.MarkDef(type='tick', 
                                  size=size, 
                                  opacity=1, 
                                  color='white')
        white_median = True
    else:
        mark_median = alt.MarkDef(type='tick', size=size, opacity=1)
        white_median = False

    # whisker
    mark_whisker = alt.MarkDef(type='rule')

    # bottom cap
    mark_bottom_cap = alt.MarkDef(type='tick', size=size/4)

    # top cap
    mark_top_cap = alt.MarkDef(type='tick', size=size/4)

    # outliers
    mark_outliers = alt.MarkDef(type='point')

    return (mark_box, mark_median, mark_whisker, mark_bottom_cap,
            mark_top_cap, mark_outliers, white_median)


def _parse_encoding_box(encoding, sort):
    """Parse encoding for a box plot."""
    if type(encoding) != dict:
        raise RuntimeError('`encoding` must be specified as a dict.')

    err = ("Exactly one of encoding['x'] or encoding['y'] must have" 
            + " quantitative encoding.")

    if 'x' not in encoding or 'y' not in encoding:
        raise RuntimeError("Both 'x' and 'y' must be in `encoding`.")

    x = _make_altair_encoding(encoding['x'], encoding=alt.X)
    y = _make_altair_encoding(encoding['y'], encoding=alt.Y)

    if _get_data_type(x) in 'NO':
        if _get_data_type(y) != 'Q':
            raise RuntimeError(err)
        cat = _get_column_name(x)
        val = _get_column_name(y)

        x = _make_altair_encoding(x,
                    encoding=alt.X, 
                    scale=_make_altair_encoding(x._kwds['scale'],
                                                encoding=alt.Scale, 
                                                domain=sort))

        if y.title == Undefined:
            y.title = val

        horizontal = False
        color = _make_color_encoding_box_jitter(encoding, cat, sort)

        # Box
        y.shorthand = 'bottom:Q'
        y2 = _make_altair_encoding('top:Q', encoding=alt.Y2)
        encoding_box = dict(x=x, y=y, y2=y2, color=color)

        # Median
        y = y.copy(deep=True)
        y.shorthand = 'middle:Q'
        encoding_median = dict(x=x, y=y, color=color)

        # Whisker
        y = y.copy(deep=True)
        y2 = y2.copy(deep=True)
        y.shorthand = 'bottom_whisker:Q'
        y2.shorthand = 'top_whisker:Q'
        encoding_whisker = dict(x=x, y=y, y2=y2, color=color)

        # bottom cap
        encoding_bottom_cap = dict(x=x, y=y, color=color)

        # top cap
        y = y.copy(deep=True)
        y.shorthand = 'top_whisker:Q'
        encoding_top_cap = dict(x=x, y=y, color=color)

        # Outliers
        y = y.copy(deep=True)
        y.shorthand = val + ':Q'
        encoding_outliers = dict(x=x, y=y, color=color)
    elif _get_data_type(y) in 'NO':
        if _get_data_type(x) != 'Q':
            raise RuntimeError(err)
        cat = _get_column_name(y)
        val = _get_column_name(x)

        y = _make_altair_encoding(y,
                    encoding=alt.Y, 
                    scale=_make_altair_encoding(y._kwds['scale'],
                                                encoding=alt.Scale, 
                                                domain=sort))

        if x.title == Undefined:
            x.title = val

        horizontal = True
        color = _make_color_encoding_box_jitter(encoding, cat, sort)

        # Box
        x.shorthand = 'bottom:Q'
        x2 = _make_altair_encoding('top:Q', encoding=alt.X2)
        encoding_box = dict(x=x, x2=x2, y=y, color=color)

        # Median
        x = x.copy(deep=True)
        x.shorthand = 'middle:Q'
        encoding_median = dict(x=x, y=y, color=color)

        # Whisker
        x = x.copy(deep=True)
        x2 = x2.copy(deep=True)
        x.shorthand = 'bottom_whisker:Q'
        x2.shorthand = 'top_whisker:Q'
        encoding_whisker = dict(x=x, x2=x2, y=y, color=color)

        # bottom cap
        encoding_bottom_cap = dict(x=x, y=y, color=color)

        # top cap
        x = x.copy(deep=True)
        x.shorthand = 'top_whisker:Q'
        encoding_top_cap = dict(x=x, y=y, color=color)

        # Outliers
        x = x.copy(deep=True)
        x.shorthand = val + ':Q'
        encoding_outliers = dict(x=x, y=y, color=color)

    return (encoding_box, encoding_median, encoding_whisker,
            encoding_bottom_cap, encoding_top_cap, encoding_outliers,
            cat, val, horizontal)


def _dimensions_box(data, cat, height, width, horizontal):
    """Determine dimensions of plot for box plot."""
    # Number of boxes
    if cat is None:
        n_boxes = 1
    else:
        n_boxes = len(data[cat].unique())

    # Set default heights and widths, also of bars
    if width == Undefined:
        if horizontal:
            width = 400
        else:
            width = 200
    if height is Undefined:
        if horizontal:
            height = 200
        else:
            height = 300

    if horizontal:
        size = height*0.7 / n_boxes
    else:
        size = width*0.7 / n_boxes

    return height, width, size


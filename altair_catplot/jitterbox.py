import numpy as np
import pandas as pd

import altair as alt
from altair.utils.schemapi import Undefined, UndefinedType

from .jitter import _jitter_plot

from .box import (_box_dataframe, _parse_mark_box)

from .utils import (_check_catplot_transform,
                    _check_catplot_sort,
                    _check_mark,
                    _make_altair_encoding,
                    _get_column_name,
                    _get_data_type,
                    _make_color_encoding_ecdf,
                    _make_color_encoding_box_jitter)


def _jitterbox_plot(data, height, width, mark, box_mark, whisker_mark,
                    encoding, jitter_width, sort, **kwargs):
    """Generate a jitter-box plot with Altair.
    """

    jitter = _jitter_plot(data, height, width, mark, encoding, jitter_width, 
                          sort, **kwargs)

    box = _box_plot_q(data, height, width, mark, box_mark, whisker_mark, 
                      encoding, sort, jitter_width, **kwargs)

    return alt.layer(jitter, box)


def _box_plot_q(data, height, width, mark, box_mark, whisker_mark, encoding,
                sort, jitter_width, **kwargs):
    """Generate a box plot with Altair."""
    # Encodings
    (encoding_box, encoding_median, encoding_bottom_whisker,
     encoding_top_whisker, encoding_bottom_cap, encoding_top_cap,
     cat, val, horizontal) = _parse_encoding_box_q(encoding, sort)

    _check_catplot_sort(data, cat, sort)

    (mark_box, _, mark_whisker, _, _, white_median) = _parse_mark_box(
            mark, box_mark, whisker_mark, jitter_width)

    mark_median = mark_whisker.copy(deep=True)
    if mark_box['color'] != Undefined:
        encoding_box['color'] = Undefined
        encoding_median['color'] = Undefined
    if mark_whisker['color'] != Undefined:
        encoding_top_whisker['color'] = Undefined
        encoding_bottom_whisker['color'] = Undefined
        encoding_top_cap['color'] = Undefined
        encoding_bottom_cap['color'] = Undefined
            
    # Data frame for boxes and whiskers
    df_box = _box_dataframe_q(data, cat, val, jitter_width)

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
                             mark=mark_whisker,
                             encoding=encoding_median,
                             **kwargs)
    chart_top_whisker = alt.Chart(data=df_box,
                                  width=width, 
                                  height=height,
                                  mark=mark_whisker,
                                  encoding=encoding_top_whisker,
                                  **kwargs)
    chart_bottom_whisker = alt.Chart(data=df_box,
                                     width=width, 
                                     height=height,
                                     mark=mark_whisker,
                                     encoding=encoding_bottom_whisker,
                                     **kwargs)
    chart_bottom_cap = alt.Chart(data=df_box,
                                 width=width, 
                                 height=height,
                                 mark=mark_whisker,
                                 encoding=encoding_bottom_cap,
                                 **kwargs)
    chart_top_cap = alt.Chart(data=df_box,
                              width=width, 
                              height=height,
                              mark=mark_whisker,
                              encoding=encoding_top_cap,
                              **kwargs)

    print(encoding_bottom_cap)
    print()

    return alt.layer(chart_bottom_whisker,
                     chart_top_whisker,
                     chart_box,
                     chart_median, 
                     chart_bottom_cap,
                     chart_top_cap)

def _box_dataframe_q(data, cat, val, jitter_width):
    """Build dataframe for box plot"""
    df_box, _ = _box_dataframe(data, cat, val)
    df_box = df_box.reset_index().rename(
                        columns={'index': 'nominal_axis_value'})
    df_box['left'] = df_box['nominal_axis_value'] - jitter_width
    df_box['right'] = df_box['nominal_axis_value'] + jitter_width
    df_box['cap_left'] = df_box['nominal_axis_value'] - jitter_width/4
    df_box['cap_right'] = df_box['nominal_axis_value'] + jitter_width/4

    return df_box

def _parse_encoding_box_q(encoding, sort):
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
        if 'color' in encoding and encoding['color'] != Undefined:
            color = _make_color_encoding_box_jitter(encoding, cat, sort)
        else:
            color = Undefined

        # The box
        x = x.copy(deep=True)
        y = y.copy(deep=True)
        x2 = _make_altair_encoding('right:Q', encoding=alt.X2)
        y2 = _make_altair_encoding('top:Q', encoding=alt.Y2)
        x.shorthand = 'left:Q'
        y.shorthand = 'bottom:Q'
        encoding_box = dict(x=x, x2=x2, y=y, y2=y2, color=color)

        # Median
        y = y.copy(deep=True)
        y.shorthand = 'middle:Q'
        encoding_median = dict(x=x, x2=x2, y=y, color=color)

        # Bottom whisker
        x = x.copy(deep=True)
        y = y.copy(deep=True)
        y2 = y2.copy(deep=True)
        x.shorthand = 'nominal_axis_value:Q'
        y.shorthand = 'bottom_whisker:Q'
        y2.shorthand = 'bottom:Q'
        encoding_bottom_whisker = dict(x=x, y=y, y2=y2, color=color)

        # Top whisker
        y = y.copy(deep=True)
        y2 = y2.copy(deep=True)
        y.shorthand = 'top:Q'
        y2.shorthand = 'top_whisker:Q'
        encoding_top_whisker = dict(x=x, y=y, y2=y2, color=color)

        # bottom cap
        x = _make_altair_encoding('cap_left:Q', encoding=alt.X)
        x2 = _make_altair_encoding('cap_right:Q', encoding=alt.X2)
        y = y.copy(deep=True)
        y.shorthand = 'bottom_whisker:Q'
        encoding_bottom_cap = dict(x=x, x2=x2, y=y, color=color)

        # top cap
        y = y.copy(deep=True)
        y.shorthand = 'top_whisker:Q'
        encoding_top_cap = dict(x=x, x2=x2, y=y, color=color)
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
        if 'color' in encoding and encoding['color'] != Undefined:
            color = _make_color_encoding_box_jitter(encoding, cat, sort)
        else:
            color = Undefined

        # Box
        x.shorthand = 'bottom:Q'
        x2 = _make_altair_encoding('top:Q', encoding=alt.X2)
        encoding_box = dict(x=x, x2=x2, y=y, color=color)

        # Median
        x = x.copy(deep=True)
        x.shorthand = 'middle:Q'
        encoding_median = dict(x=x, y=y, color=color)

        # Bottom whisker
        x = x.copy(deep=True)
        x2 = x2.copy(deep=True)
        x.shorthand = 'bottom_whisker:Q'
        x2.shorthand = 'bottom:Q'
        encoding_bottom_whisker = dict(x=x, x2=x2, y=y, color=color)

        # Top whisker
        x = x.copy(deep=True)
        x2 = x2.copy(deep=True)
        x.shorthand = 'top:Q'
        x2.shorthand = 'top_whisker:Q'
        encoding_top_whisker = dict(x=x, x2=x2, y=y, color=color)

        # bottom cap
        x = x.copy(deep=True)
        x.shorthand = 'bottom_whisker:Q'
        encoding_bottom_cap = dict(x=x, y=y, color=color)

        # top cap
        x = x.copy(deep=True)
        x.shorthand = 'top_whisker:Q'
        encoding_top_cap = dict(x=x, y=y, color=color)

        # Outliers
        x = x.copy(deep=True)
        x.shorthand = val + ':Q'
        encoding_outliers = dict(x=x, y=y, color=color)

    return (encoding_box, encoding_median, encoding_bottom_whisker,
            encoding_top_whisker, encoding_bottom_cap, encoding_top_cap,
            cat, val, horizontal)


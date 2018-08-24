import numpy as np
import pandas as pd

import altair as alt
from altair.utils.schemapi import Undefined, UndefinedType

from .jitter import (_jitter_plot, _jitter_sort)

from .box import (_box_dataframe, _parse_mark_box)

from .utils import (_check_catplot_transform,
                    _check_catplot_sort,
                    _check_mark,
                    _make_altair_encoding,
                    _get_column_name,
                    _get_data_type,
                    _make_color_encoding)


def _jitterbox_plot(data, height, width, mark, box_mark, whisker_mark,
                    encoding, jitter_width, sort, box_overlay, **kwargs):
    """Generate a jitter-box plot with Altair.
    """

    jitter = _jitter_plot(data, height, width, mark, encoding, jitter_width, 
                          sort, **kwargs)

    box = _box_plot_q(data, height, width, mark, box_mark, whisker_mark, 
                      encoding, sort, jitter_width, **kwargs)

    if box_overlay:
        return alt.layer(jitter, box)
    else:
        return alt.layer(box, jitter)


def _box_plot_q(data, height, width, mark, box_mark, whisker_mark, encoding,
                sort, jitter_width, **kwargs):
    """Generate a box plot with Altair."""

    # Set up default marks
    if box_mark == Undefined or box_mark is None:
        box_mark = dict(color='lightgray', filled=True)
    if whisker_mark == Undefined or whisker_mark is None:
        whisker_mark = dict(color='lightgray')

    # Adjust as needed
    (mark_box, _, mark_whisker, _, _, white_median) = _parse_mark_box(
            mark, box_mark, whisker_mark, jitter_width)

    # Encodings
    (encoding_box, encoding_median, encoding_bottom_whisker,
     encoding_top_whisker, encoding_bottom_cap, encoding_top_cap,
     cat, val, horizontal) = _parse_encoding_box_q(encoding, sort, white_median)

    sort = _jitter_sort(data, cat, sort, horizontal)

    mark_median = mark_whisker.copy(deep=True)
    if white_median:
        mark_median['color'] = 'white'
        mark_median['opacity'] = 1
    if mark_box['color'] != Undefined:
        encoding_box['color'] = Undefined
        encoding_median['color'] = Undefined
    if mark_whisker['color'] != Undefined:
        encoding_top_whisker['color'] = Undefined
        encoding_bottom_whisker['color'] = Undefined
        encoding_top_cap['color'] = Undefined
        encoding_bottom_cap['color'] = Undefined
            
    # Data frame for boxes and whiskers
    df_box = _box_dataframe_q(data, cat, val, jitter_width, sort)

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

    return alt.layer(chart_bottom_whisker,
                     chart_top_whisker,
                     chart_box,
                     chart_median, 
                     chart_bottom_cap,
                     chart_top_cap)

def _box_dataframe_q(data, cat, val, jitter_width, sort):
    """Build dataframe for box plot"""
    df_box, _ = _box_dataframe(data, cat, val)
    df_box = df_box.reset_index().rename(
                        columns={'index': 'nominal_axis_value'})

    if sort == Undefined:
        centers = pd.Categorical(df_box[cat]).codes
    else:
        cats = list(pd.Categorical(df_box[cat]))
        centers = np.array([sort.index(c) for c in cats], dtype=float)

    df_box['nominal_axis_value'] = centers


    df_box['left'] = df_box['nominal_axis_value'] - jitter_width
    df_box['right'] = df_box['nominal_axis_value'] + jitter_width
    df_box['cap_left'] = df_box['nominal_axis_value'] - jitter_width/4
    df_box['cap_right'] = df_box['nominal_axis_value'] + jitter_width/4

    return df_box

def _parse_encoding_box_q(encoding, sort, white_median):
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
        color = _make_color_encoding(encoding, cat, sort)

        x = _make_altair_encoding(x,
                    encoding=alt.X, 
                    scale=_make_altair_encoding(x._kwds['scale'],
                                                encoding=alt.Scale, 
                                                domain=sort))

        if y.title == Undefined:
            y.title = val

        horizontal = False

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
        if white_median:
            encoding_median = dict(x=x, x2=x2, y=y)
        else:
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
        color = _make_color_encoding(encoding, cat, sort)

        y = _make_altair_encoding(y,
                    encoding=alt.Y, 
                    scale=_make_altair_encoding(y._kwds['scale'],
                                                encoding=alt.Scale, 
                                                domain=sort))

        if x.title == Undefined:
            x.title = val

        horizontal = True

        # The box
        x = x.copy(deep=True)
        y = y.copy(deep=True)
        x2 = _make_altair_encoding('top:Q', encoding=alt.X2)
        y2 = _make_altair_encoding('right:Q', encoding=alt.Y2)
        x.shorthand = 'bottom:Q'
        y.shorthand = 'left:Q'
        encoding_box = dict(x=x, x2=x2, y=y, y2=y2, color=color)

        # Median
        x = x.copy(deep=True)
        x.shorthand = 'middle:Q'
        if white_median:
            encoding_median = dict(x=x, y=y, y2=y2)
        else:
            encoding_median = dict(x=x, y=y, y2=y2, color=color)

        # Bottom whisker
        x = x.copy(deep=True)
        y = y.copy(deep=True)
        x2 = x2.copy(deep=True)
        x.shorthand = 'bottom_whisker:Q'
        x2.shorthand = 'bottom:Q'
        y.shorthand = 'nominal_axis_value:Q'
        encoding_bottom_whisker = dict(x=x, x2=x2, y=y, color=color)

        # Top whisker
        x = x.copy(deep=True)
        x2 = x2.copy(deep=True)
        x.shorthand = 'top:Q'
        x2.shorthand = 'top_whisker:Q'
        encoding_top_whisker = dict(x=x, x2=x2, y=y, color=color)

        # bottom cap
        y = _make_altair_encoding('cap_left:Q', encoding=alt.Y)
        y2 = _make_altair_encoding('cap_right:Q', encoding=alt.Y2)
        x = x.copy(deep=True)
        x.shorthand = 'bottom_whisker:Q'
        encoding_bottom_cap = dict(x=x, y=y, y2=y2, color=color)

        # top cap
        x = x.copy(deep=True)
        x.shorthand = 'top_whisker:Q'
        encoding_top_cap = dict(x=x, y=y, y2=y2, color=color)

    return (encoding_box, encoding_median, encoding_bottom_whisker,
            encoding_top_whisker, encoding_bottom_cap, encoding_top_cap,
            cat, val, horizontal)


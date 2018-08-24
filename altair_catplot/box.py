import numpy as np
import pandas as pd

import altair as alt
from altair.utils.schemapi import Undefined, UndefinedType

from .utils import (_check_catplot_transform,
                    _check_catplot_sort,
                    _check_mark,
                    _make_altair_encoding,
                    _get_column_name,
                    _get_data_type,
                    _make_color_encoding)

def _box_plot(data, height, width, mark, box_mark, whisker_mark, encoding,
              sort, **kwargs):
    """Generate a box plot with Altair."""
    # Encodings
    (encoding_box, encoding_median, encoding_bottom_whisker,
     encoding_top_whisker, encoding_bottom_cap, encoding_top_cap, 
     encoding_outliers, cat, val, 
     horizontal) = _parse_encoding_box(encoding, sort)

    _check_catplot_sort(data, cat, sort)

    # Chart dimensions
    height, width, size = _dimensions_box(data, 
                                          cat, 
                                          height, 
                                          width, 
                                          horizontal)

    # Marks
    (mark_box, mark_median, mark_whisker, mark_cap, mark_outliers,
     white_median) = _parse_mark_box(mark, box_mark, whisker_mark, size)

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
                                 mark=mark_cap,
                                 encoding=encoding_bottom_cap,
                                 **kwargs)
    chart_top_cap = alt.Chart(data=df_box,
                              width=width, 
                              height=height,
                              mark=mark_cap,
                              encoding=encoding_top_cap,
                              **kwargs)
    chart_outliers = alt.Chart(data=df_outliers,
                               width=width, 
                               height=height,
                               mark=mark_outliers,
                               encoding=encoding_outliers,
                               **kwargs)

    return alt.layer(chart_bottom_whisker,
                     chart_top_whisker,
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


def _parse_mark_box(mark, box_mark, whisker_mark, size):
    """Parse mark for box plot."""

    # The box
    if box_mark is None or box_mark == Undefined:
        mark_box = alt.MarkDef(type='bar', size=size)
    elif type(box_mark) != dict:
        raise RuntimeError("`box_mark` must be a dict or None.")
    else:
        if 'type' in box_mark:
            if box_mark['type'] != 'bar':
                raise RuntimeError(
                            "`box_mark['type']` must be 'bar' for box plot.")
            del box_mark['type']
        if 'size' in box_mark:
            size = box_mark['size']
            del box_mark['size']
        mark_box = alt.MarkDef(type='bar', size=size, **box_mark)

    # Check whiskers
    strokeWidth = Undefined
    if whisker_mark is None or whisker_mark == Undefined:
        mark_whisker = alt.MarkDef(type='rule')
        mark_cap = alt.MarkDef(type='tick', size=size/4, opacity=1)
        whisker_mark = dict()
    elif type(whisker_mark) != dict:
        raise RuntimeError("`whisker_mark` must be a dict or None.")
    else:
        if 'type' in whisker_mark:
            if whisker_mark['type'] != 'rule':
                raise RuntimeError(
                        "`whisker_mark['type']` must be 'rule' for box plot.")
            del whisker_mark['type']
        if 'size' in whisker_mark:
            strokeWidth = whisker_mark['size']
            del whisker_mark['size']
        if 'strokeWidth' in whisker_mark:
            strokeWidth = whisker_mark['strokeWidth']
            del whisker_mark['strokeWidth']
        if 'opacity' not in whisker_mark:
            whisker_mark['opacity'] = 1
        mark_whisker = alt.MarkDef(type='rule',
                                   strokeWidth=strokeWidth, 
                                   **whisker_mark)
        mark_cap = alt.MarkDef(type='tick',
                               thickness=strokeWidth, 
                               size=size/4,
                               **whisker_mark)

    # median
    if mark_box.filled == Undefined or mark_box.filled:
        if 'opacity' in whisker_mark:
            del whisker_mark['opacity']
        if 'color' in whisker_mark:
            del whisker_mark['color']
        mark_median = alt.MarkDef(type='tick', 
                                  size=size,
                                  thickness=strokeWidth, 
                                  color='white',
                                  opacity=1,
                                  **whisker_mark)
        white_median = True
    else:
        if 'opacity' not in whisker_mark:
            whisker_mark['opacity'] = 1
        mark_median = alt.MarkDef(type='tick', 
                                  size=size, 
                                  thickness=strokeWidth,
                                  **whisker_mark)
        white_median = False

    if mark == Undefined or mark is None:
        mark = 'point'
    if mark in ['point', 'circle', 'square']:
        mark_outliers =  alt.MarkDef(type=mark)
    elif type(mark) != dict:
        raise RuntimeError("""`mark` must be a dict or be one of:
                'point'
                'circle'
                'square'""")
    else:  
        mark_outliers = alt.MarkDef(**mark)

    return (mark_box, mark_median, mark_whisker, mark_cap,
            mark_outliers, white_median)


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

    extra_encodings = {key: val for key, val in encoding.items()
                            if key not in ['x', 'y', 'x2', 'y2', 'color']}

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

        # Box
        y.shorthand = 'bottom:Q'
        y2 = _make_altair_encoding('top:Q', encoding=alt.Y2)
        encoding_box = dict(x=x, y=y, y2=y2, color=color, **extra_encodings)

        # Median
        y = y.copy(deep=True)
        y.shorthand = 'middle:Q'
        encoding_median = dict(x=x, y=y, color=color, **extra_encodings)

        # Bottom whisker
        y = y.copy(deep=True)
        y2 = y2.copy(deep=True)
        y.shorthand = 'bottom_whisker:Q'
        y2.shorthand = 'bottom:Q'
        encoding_bottom_whisker = dict(x=x, y=y, y2=y2, color=color, 
                                       **extra_encodings)

        # Top whisker
        y = y.copy(deep=True)
        y2 = y2.copy(deep=True)
        y.shorthand = 'top:Q'
        y2.shorthand = 'top_whisker:Q'
        encoding_top_whisker = dict(x=x, y=y, y2=y2, color=color,
                                    **extra_encodings)

        # bottom cap
        y = y.copy(deep=True)
        y.shorthand = 'bottom_whisker:Q'
        encoding_bottom_cap = dict(x=x, y=y, color=color, **extra_encodings)

        # top cap
        y = y.copy(deep=True)
        y.shorthand = 'top_whisker:Q'
        encoding_top_cap = dict(x=x, y=y, color=color, **extra_encodings)

        # Outliers
        y = y.copy(deep=True)
        y.shorthand = val + ':Q'
        encoding_outliers = dict(x=x, y=y, color=color, **extra_encodings)
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

        # Box
        x.shorthand = 'bottom:Q'
        x2 = _make_altair_encoding('top:Q', encoding=alt.X2)
        encoding_box = dict(x=x, x2=x2, y=y, color=color, **extra_encodings)

        # Median
        x = x.copy(deep=True)
        x.shorthand = 'middle:Q'
        encoding_median = dict(x=x, y=y, color=color, **extra_encodings)

        # Bottom whisker
        x = x.copy(deep=True)
        x2 = x2.copy(deep=True)
        x.shorthand = 'bottom_whisker:Q'
        x2.shorthand = 'bottom:Q'
        encoding_bottom_whisker = dict(x=x, x2=x2, y=y, color=color, 
                                       **extra_encodings)

        # Top whisker
        x = x.copy(deep=True)
        x2 = x2.copy(deep=True)
        x.shorthand = 'top:Q'
        x2.shorthand = 'top_whisker:Q'
        encoding_top_whisker = dict(x=x, x2=x2, y=y, color=color, 
                                    **extra_encodings)

        # bottom cap
        x = x.copy(deep=True)
        x.shorthand = 'bottom_whisker:Q'
        encoding_bottom_cap = dict(x=x, y=y, color=color, **extra_encodings)

        # top cap
        x = x.copy(deep=True)
        x.shorthand = 'top_whisker:Q'
        encoding_top_cap = dict(x=x, y=y, color=color, **extra_encodings)

        # Outliers
        x = x.copy(deep=True)
        x.shorthand = val + ':Q'
        encoding_outliers = dict(x=x, y=y, color=color, **extra_encodings)

    return (encoding_box, encoding_median, encoding_bottom_whisker,
            encoding_top_whisker, encoding_bottom_cap, encoding_top_cap,
            encoding_outliers, cat, val, horizontal)


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


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


def _ecdf_plot(data, height, width, mark, encoding, complementary, 
               colored, sort, **kwargs):
    """Make an ECDF using Altair."""
    mark = _parse_mark_ecdf(mark)

    encoding, cat, val = _parse_encoding_ecdf(encoding, complementary, sort)

    _check_catplot_sort(data, cat, sort)

    formal = mark.type=='line'

    if formal and colored:
        raise RuntimeError('Only point-ECDFs are allowed for `colored_ecdf`.')

    df = _ecdf_dataframe(data,
                         val=val, 
                         cat=cat, 
                         formal=formal, 
                         complementary=complementary,
                         colored=colored)

    return alt.Chart(data=df, 
                     height=height,
                     width=width, 
                     mark=mark, 
                     encoding=encoding,
                     **kwargs)

def _ecdf_vals(data, formal=False, x_min=None, x_max=None):
    """Get x, y, values of an ECDF for plotting.

    Parameters
    ----------
    data : ndarray
        One dimensional Numpy array with data.
    formal : bool, default False
        If True, generate x and y values for formal ECDF (staircase). If
        False, generate x and y values for ECDF as dots.
    x_min : float, 'infer', or None
        Minimum value of x to plot. If 'infer', use a 5% buffer. Ignored
        if `formal` is False.
    x_max : float, 'infer', or None
        Maximum value of x to plot. If 'infer', use a 5% buffer. Ignored
        if `formal` is False.

    Returns
    -------
    x : ndarray
        x-values for plot
    y : ndarray
        y-values for plot
    """
    x = np.sort(data)
    y = np.arange(1, len(data)+1) / len(data)

    if formal:
        return _to_formal(x, y)
    else:
        return x, y


def _to_formal(x, y, x_min, x_max):
    """Convert to formal ECDF."""
    # Set up output arrays
    x_formal = np.empty(2*len(x))
    y_formal = np.empty(2*len(x))

    # y-values for steps
    y_formal[0] = 0
    y_formal[1::2] = y
    y_formal[2::2] = y[:-1]

    # x- values for steps
    x_formal[::2] = x
    x_formal[1::2] = x

    # Put lines at y=0
    if x_min is not None:
        if x_min == 'infer':
            x_min = x.min() - (x.max() - x.min())*0.05
        elif x_min > x.min():
            raise RuntimeError('x_min > x.min().')
        x_formal = np.concatenate(((x_min,), x_formal))
        y_formal = np.concatenate(((0,), y_formal))

    # Put lines at y=y.max()
    if x_max is not None:
        if x_max == 'infer':
            x_max = x.max() + (x.max() - x.min())*0.05
        elif x_max < x.max():
            raise RuntimeError('x_max < x.max().')
        x_formal = np.concatenate((x_formal, (x_max,)))
        y_formal = np.concatenate((y_formal, (y.max(),)))

    return x_formal, y_formal


def _ecdf_y(data, complementary=False):
    """Give y-values of an ECDF for an unsorted column in a data frame.
    
    Parameters
    ----------
    data : Pandas Series
        Series (or column of a DataFrame) from which to generate ECDF
        values
    complementary : bool, default False
        If True, give the ECCDF values.

    Returns
    -------
    output : Pandas Series
        Corresponding y-values for an ECDF when plotted with dots.

    Notes
    -----
    .. This only works for plotting an ECDF with points, not for formal
       ECDFs
    """
    if complementary:
        return 1 - data.rank(method='first') / len(data) + 1 / len(data)
    else:
        return data.rank(method='first') / len(data)


def _point_ecdf_dataframe(data=None, val=None, cat=None, complementary=False,
                          colored=False):
    """DataFrame for making point-wise ECDF."""
    df = data.copy()

    if complementary:
        col = '__ECCDF'
    else:
        col = '__ECDF'

    if cat is None or colored:
        df[col] = _ecdf_y(df[val], complementary)
    else:
        df[col] = df.groupby(cat)[val].transform(_ecdf_y, complementary)

    return df


def _formal_ecdf_dataframe(data=None, val=None, cat=None,
                           complementary=False):
    """DataFrame for making formal ECDF."""
     # Determine range for plot
    data_min = data[val].min()
    data_max = data[val].max()
    x_min = data_min - (data_max - data_min) * 0.05
    x_max = data_max + (data_max - data_min) * 0.05

    if cat is None:
        x_ecdf, y_ecdf = _ecdf_vals(data[val].values,
                                    formal=True,
                                    x_min=x_min, 
                                    x_max=x_max)
        return pd.DataFrame({val: x_ecdf, '__ECDF': y_ecdf})
    else:
        grouped = data.groupby(cat)
        df_list = []
        for g in grouped:
            if type(g[0]) == tuple:
                cat_str = ', '.join([str(c) for c in g[0]])
            else:
                cat_str = g[0]

            x_ecdf, y_ecdf = _ecdf_vals(g[1][val],
                                       formal=True,
                                       x_min=x_min, 
                                       x_max=x_max)

            if complementary:
                new_df = pd.DataFrame(data={cat: [cat_str]*len(x_ecdf),
                                            val: x_ecdf, 
                                            '__ECCDF': 1 - y_ecdf})
            else:
                new_df = pd.DataFrame(data={cat: [cat_str]*len(x_ecdf),
                                            val: x_ecdf, 
                                            '__ECDF': y_ecdf})
            df_list.append(new_df)

        return pd.concat(df_list, ignore_index=True, sort=False)


def _ecdf_dataframe(data=None, val=None, cat=None, formal=False,
                    complementary=False, colored=False):
    """Generate a DataFrame that can be used for plotting ECDFs.

    Parameters
    ----------
    data : Pandas DataFrame
        A tidy data frame.
    val : valid column name of Pandas DataFrame
        Column of data frame containing values to use in ECDF plot.
    cat : valid column name of Pandas DataFrame or list of column 
            names
        Column(s) of DataFrame to use for grouping the data. A unique
        set of ECDF values is made for each. If None, no groupby 
        operations are performed and a single ECDF is generated.
    formal : bool, default False
        If True, generate val and y values for formal ECDF (staircase). If
        False, generate val and y values for ECDF as dots.
    complementary : bool, default False
        If True, give the ECCDF values.       

    Returns
    -------
    output : Pandas DataFrame
        Pandas DataFrame with two or three columns.
            val : Column named for inputted `val`, data values.
            'ECDF': Values for y-values for plotting the ECDF
            cat : Keys for groups. Omitted if `cat` is None.
    """
    if data is None:
        raise RuntimeError('`data` must be specified.')
    if val is None:
        raise RuntimeError('`val` must be specified.')

    if formal:
        return _formal_ecdf_dataframe(data=data,
                                      val=val, 
                                      cat=cat, 
                                      complementary=complementary)
    else:
        return _point_ecdf_dataframe(data=data, 
                                     val=val, 
                                     cat=cat, 
                                     complementary=complementary,
                                     colored=colored)


def _is_ecdf_axis(x):
    if x is None:
        return True

    if x in ['ecdf', 'ECDF', 'ecdf:Q', 'ECDF:Q', 
             'eccdf', 'ECCDF', 'eccdf:Q', 'ECCDF:Q']:
        return True

    if (    (isinstance(x, alt.X) or isinstance(x, alt.Y)) 
        and _get_column_name(x) in ['ecdf', 'ECDF', 'eccdf', 'ECCDF']):
        return True

    return False


def _make_xy_encoding_ecdf(encoding, complementary):
    """Make x and y encodings for an ECDF plot"""
    if complementary:
        title = 'ECCDF'
    else:
        title = 'ECDF'

    err = ("Exactly one of encoding['x'] or encoding['y'] must have"
           + " quantitative encoding and one must be designated as ECDF.")

    if 'x' not in encoding or _is_ecdf_axis(encoding['x']):
        if 'y' not in encoding or _is_ecdf_axis(encoding['y']):
            raise RuntimeError(err)

        if 'x' in encoding:
            x = _make_altair_encoding(encoding['x'], alt.X)
            x.shorthand = '__' + title + ':Q'
        else:
            x = _make_altair_encoding('__' + title + ':Q', alt.X)
        if x.title is Undefined:
            x.title = title
        y = _make_altair_encoding(encoding['y'], alt.Y)
        val = _get_column_name(y)
    elif 'y' not in encoding or _is_ecdf_axis(encoding['y']):
        if 'y' in encoding:
            y = _make_altair_encoding(encoding['y'], alt.Y)
            y.shorthand = '__' + title + ':Q'
        else:
            y = _make_altair_encoding('__' + title + ':Q', alt.Y)
        if y.title is Undefined:
            y.title = title
        x = _make_altair_encoding(encoding['x'], alt.X)
        val = _get_column_name(x)
    else:
        raise RuntimeError(err)

    return x, y, val


def _parse_encoding_ecdf(encoding, complementary, sort):
    """Parse encoding for ECDF."""
    if type(encoding) != dict:
        raise RuntimeError('`encoding` must be specified as a dict.')

    x, y, val = _make_xy_encoding_ecdf(encoding, complementary)

    # Need to run through color spec twice to get cat
    color = _make_color_encoding(encoding, None, sort)
    if color == Undefined:
        cat = None
    else:
        cat = _get_column_name(color)
        color = _make_color_encoding(encoding, cat, sort)

    # Build encoding for export
    encoding = {key: item for key, item in encoding.items() 
                if key not in ['x', 'y', 'color']}
    encoding['x'] = x
    encoding['y'] = y

    if cat is not None:
        encoding['color'] = color

    encoding['order'] = alt.Order(val+':Q', sort='ascending')

    return encoding, cat, val


def _parse_mark_ecdf(mark):
    """Parse mark for ECDF."""
    if mark in ['point', 'circle', 'square', 'line']:
        return alt.MarkDef(type=mark)

    if type(mark) != dict:
        raise RuntimeError("""`mark` must be a dict or be one of:
                'point'
                'circle'
                'square'
                'line'""")

    return alt.MarkDef(**mark)
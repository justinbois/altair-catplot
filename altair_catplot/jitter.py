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


def _jitter_plot(data, height, width, mark, encoding, jitter_width, sort,
                 **kwargs):
    """Generate a jitter plot with Altair.
    """
    encoding_tuple = _parse_encoding_jitter(encoding, data, sort)
    (encoding, encoding_text, cat, val, 
      nominal_axis_values, horizontal, zero) = encoding_tuple

    _check_catplot_sort(data, cat, sort)

    sort = _jitter_sort(data, cat, sort, horizontal)

    mark_jitter, mark_text = _parse_mark_jitter(mark, horizontal)

    df, df_text = _jitter_dataframe(data, 
                                    val, 
                                    cat, 
                                    jitter_width, 
                                    nominal_axis_values,
                                    sort, 
                                    zero)

    chart_jitter = alt.Chart(data=df,
                             width=width,
                             height=height,
                             mark=mark_jitter,
                             encoding=encoding,
                             **kwargs)

    chart_text = alt.Chart(data=df_text,
                           width=width,
                           height=height,
                           mark=mark_text,
                           encoding=encoding_text)

    return alt.layer(chart_jitter, chart_text)


def _jitter_sort(data, cat, sort, horizontal):
    """Generate sort 
    """
    if sort == Undefined:
        sort = list(data[cat].unique())

    if horizontal:
        return list(reversed(sort))
    else:
        return sort


def _jitter(x, sort, jitter_width):
    """Make x-coordinates for a jitter plot."""
    if sort == Undefined:
        centers = pd.Categorical(x).codes
    else:
        cats = list(pd.Categorical(x))
        centers = np.array([sort.index(c) for c in cats], dtype=float)

    return (centers
            + np.random.uniform(low=-jitter_width,
                                high=jitter_width,
                                size=len(x)))


def _check_altair_jitter_input(data, height, width, mark, encoding,
                               jitter_width):
    if mark not in ['point', 'circle', 'square', 'tick']:
        raise RuntimeError("""Invalid `mark`. 
Allowed values are ['point', 'circle', 'square', 'tick'].""")
    if data is None:
        raise RuntimeError('`data` must be specified.')
    if not (0 <= jitter_width <= 0.5):
        raise RuntimeError('Must have `jitter_width` between 0 and 0.5.')


def _jitter_domain(data, val, zero, pad=0.05):
    """Determine domain for jitter plot"""
    if zero:
        return [0, Undefined]
    else:
        data_range = data[val].max() - data[val].min()
        return [data[val].min() - pad * data_range,
                data[val].max() + pad * data_range]


def _jitter_dataframe(data, val, cat, jitter_width, nominal_axis_values,
                      sort, zero):
    """DataFrame for making jitter plots."""
    df = data.copy()

    if cat is None:
        df['__jitter'] = _jitter(np.zeros(len(df)), sort, jitter_width)
    else:
        df['__jitter'] = _jitter(df[cat], sort, jitter_width)

    if zero:
        min_val = 0
    else:
        min_val = _jitter_domain(df, val, zero, pad=0.05)[0]

    # Make text data frame
    if sort == Undefined:
        text = df[cat].unique()
    else:
        cats = list(pd.Categorical(df[cat]))
        text = [cats[cats.index(s)] for s in sort]

    df_text = pd.DataFrame(data={val: [min_val]*len(nominal_axis_values),
                                 '__jitter': nominal_axis_values,
                                 'text': text})

    return df, df_text


def _parse_encoding_jitter(encoding, data, sort):
    """Parse encoding for a jitter plot."""
    if type(encoding) != dict:
        raise RuntimeError('`encoding` must be specified as a dict.')

    extra_encodings = {key: val for key, val in encoding.items()
                            if key not in ['x', 'y', 'color']}

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

        val_axis = 'y'

        nominal_axis_values = list(range(len(data[cat].unique())))

        if y.title == Undefined:
            y.title = val

        color = _make_color_encoding(encoding, cat, sort)

        encoding = dict(x=_make_altair_encoding(
                            '__jitter:Q',
                            encoding=alt.X, 
                            axis=alt.Axis(title=None,
                                          labels=False,
                                          values=nominal_axis_values,
                                          grid=False)),
                        y=y, 
                        color=color,
                        **extra_encodings)
        encoding_text = dict(x=_make_altair_encoding(
                                '__jitter:Q',
                                encoding=alt.X, 
                                axis=alt.Axis(title=None,
                                              labels=False,
                                              values=nominal_axis_values,
                                              grid=False)),
                             y=y,
                             text=alt.Text('text:N'))
    elif _get_data_type(y) in 'NO':
        if _get_data_type(x) != 'Q':
            raise RuntimeError(err)
        cat = _get_column_name(y)
        val = _get_column_name(x)

        val_axis = 'x'

        nominal_axis_values = list(range(len(data[cat].unique())))

        if x.title == Undefined:
            x.title = val

        color = _make_color_encoding(encoding, cat, sort)

        encoding = dict(x=x,
                        y=_make_altair_encoding(
                            '__jitter:Q',
                            encoding=alt.Y, 
                            axis=alt.Axis(title=None,
                                          labels=False,
                                          values=nominal_axis_values,
                                          grid=False)),
                        color=color,
                        **extra_encodings)
        encoding_text = dict(y=_make_altair_encoding(
                                '__jitter:Q',
                                encoding=alt.Y, 
                                axis=alt.Axis(title=None,
                                              labels=False,
                                              values=nominal_axis_values,
                                              grid=False)),
                             x=x,
                             text=alt.Text('text:N'))

    # Determine if y-axis is zeroed
    if data[val].min() < 0:
        if (encoding[val_axis]._kwds['scale'] != Undefined
            and encoding[val_axis]._kwds['scale']._kwds['zero']):
            zero = True
        else:
            zero = False
    elif (encoding[val_axis]._kwds['scale'] == Undefined
          or encoding[val_axis]._kwds['scale']._kwds['zero']):
        zero = True
    else:
        zero = False

    horizontal = val_axis == 'x'

    if not zero:
        domain = _jitter_domain(data, val, zero, pad=0.05)
        if horizontal:
            encoding['x'] = _make_altair_encoding(
                        x, 
                        encoding=alt.X, 
                        scale=alt.Scale(domain=domain, nice=False))
        else:
            encoding['y'] = _make_altair_encoding(
                        y, 
                        encoding=alt.Y, 
                        scale=alt.Scale(domain=domain, nice=False))

    return (encoding, encoding_text, cat, val, nominal_axis_values,
            horizontal, zero)


def _parse_mark_jitter(mark, horizontal):
    """Parse mark for jitter plot."""
    if mark in ['point', 'circle', 'square']:
        mark_jitter =  alt.MarkDef(type=mark)
    elif type(mark) != dict:
        raise RuntimeError("""`mark` must be a dict or be one of:
                'point'
                'circle'
                'square'""")
    else:  
        mark_jitter = alt.MarkDef(**mark)


    if horizontal:
        mark_text = alt.MarkDef(type='text',
                                baseline='middle',
                                align='right',
                                dx=-8)
    else:
        mark_text = alt.MarkDef(type='text',
                                baseline='top',
                                align='center',
                                dy=8)

    return mark_jitter, mark_text

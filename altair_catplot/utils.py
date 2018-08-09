import altair as alt
from altair.utils.schemapi import Undefined, UndefinedType

def _check_catplot_transform(transform):
    """Check to make sure transform is valid for catplot.

    Parameters
    ----------
    transform : str or list of strings
        Which transform to use. Valid entries are:
             'ecdf'
             'colored_ecdf'
             'eccdf'
             'colored_eccdf'
             'colored_ecdf'
             'box'
             'jitter'
             'swarm'
             ['box', 'jitter']
             ['box', 'swarm']
    
    Returns
    -------
    output : str of list of string
        Transform, as a sorted list if ['box', 'jitter'] or 
        ['box', swarm'].
    """
    if transform is None:
        raise RuntimeError('`transform` must be specified.')

    if type(transform) in [tuple, list]:
        transform = sorted(transform)

    if transform not in ['ecdf',
                         'colored_ecdf',
                         'eccdf',
                         'colored_eccdf',
                         'box',
                         'jitter',
                         ['box', 'jitter']]:
        raise RuntimeError("""Invalid transform. Valid possibilities are:
             'ecdf'
             'ecdf_collection'
             'colored_ecdf'
             'box'
             'jitter'
             ['box', 'jitter']""")

    return transform


def _check_catplot_sort(df, cat, sort):
    """Check to make sure sort is valid."""
    if cat is None and sort != Undefined:
        raise RuntimeError('No categorical variable was determined, so `sort` cannot be specified.')

    if sort != Undefined:
        cats = df[cat].unique()
        if sorted(sort) != sorted(list(df[cat].unique())):
            raise RuntimeError('`sort` must have an entry for every value of the categorical variable considered.')


def _check_mark(mark):
    """Check to make sure mark is valid."""
    if mark not in ['point', 'circle', 'square', 'line']:
        raise RuntimeError("""Invalid `mark`. Allowed values are: 
            'point'
            'circle'
            'square'
            'line'""")


def _make_altair_encoding(x, encoding, **kwargs):
    """Specified kwargs overwrite what was originally in the encoding."""
    if isinstance(x, encoding):
        input_kwds = {key: item for key, item in x._kwds.items() 
                                if item != Undefined}

        return encoding(**{**input_kwds, **kwargs})
    elif x is None:
        return encoding(**kwargs)
    else:
        return encoding(x, **kwargs)


def _get_column_name(x):
    """Get the name of a column from Altair specification."""
    if len(x.shorthand) > 1 and x.shorthand[-2] == ':':
        return x.shorthand[:-2]
    else:
        return x.shorthand


def _get_data_type(encoding):
    if not isinstance(encoding.type, UndefinedType):
        return var.type[0].upper()
    elif len(encoding.shorthand) > 1 and encoding.shorthand[-2] == ':':
        return encoding.shorthand[-1]
    else:
        return UndefinedType


def _make_color_encoding_ecdf(encoding, sort):
    """Make color encodings for an ECDF plot."""
    if 'color' in encoding:
        color = _make_altair_encoding(encoding['color'],
                    alt.Color, 
                    scale=_make_altair_encoding(
                                encoding['color']._kwds['scale'],
                                encoding=alt.Scale, 
                                domain=sort))
        color_data_type = _get_data_type(color)
        if color_data_type == UndefinedType:
            color = _make_altair_encoding(encoding['color'],
                                          alt.Color,
                                          type='nominal')
        cat = _get_column_name(color)
    else:
        color = None
        cat = None

    return color, cat


def _make_color_encoding_box_jitter(encoding, cat, sort):
    """Make color encodings for a box plot."""
    color, _ = _make_color_encoding_ecdf(encoding, sort=sort)
    if color is None:
        color = _make_altair_encoding(cat,
                    alt.Color, 
                    type='nominal',
                    scale=_make_altair_encoding(None,
                                                alt.Scale,
                                                domain=sort))

    return color

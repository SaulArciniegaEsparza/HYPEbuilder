"""
==============================================================================
HYPE hydrological model tools for python

Time series tools


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
==============================================================================
"""

import os as _os
import numpy as _np
import pandas as _pd


# ==============================================================================
# Time series for multiple basins
# ==============================================================================

def read_hype_basinserie(filename):
    """
    Reads a milti time series for a single basin

    :param filename:     [string] input filename
    :return:             [DataFrame]
    """

    if _os.path.exists(filename):
        data = _pd.read_csv(filename, sep='\t', skiprows=range(1, 2), index_col=[0],
                           parse_dates=[0])
        data = data.replace(to_replace=-9999, value=_np.nan)
        data.index.name = 'DATE'
        return data
    else:
        return _pd.DataFrame([])


def read_hype_timeserie(filename, subid=None):
    """
    Reads a multi basin time serie file as a DataFrame.

    :param filename:     [string] file name
    :param subid:        [int, tuple, list] basin id. If None, all basins are
                             returned as a DataFrame
    :return:             [DataFrame]
    """

    if _os.path.exists(filename):
        if subid is not None:
            if type(subid) in (int, float):
                subids = ['DATE'] + [str(int(subid))]
            elif type(subid) in (tuple, list, _np.ndarray):
                subids = ['DATE'] + [str(int(x)) for x in subid]
            else:
                subids = None
        else:
            subids = None
        data = _pd.read_csv(filename, sep='\t', index_col=[0], parse_dates=[0],
                           usecols=subids, skiprows=1)
        data = data.replace(to_replace=-9999, value=_np.nan)
        data.columns = [int(x) for x in data.columns]
        return data
    else:
        return _pd.DataFrame([])


def read_hype_xobs(varname=None):
    """
    Reads a multi basin and multi variable time serie as a DataFrame

    :param varname:      [string] variable name. If None, all variables are returned as a
                           multi index DataFrame
    :return:             [DataFrame]
    """

    if _os.path.exists(filename):
        xobs = _pd.read_csv(filename, sep='\t', skiprows=1, header=[0, 1], index_col=[0],
                            parse_dates=[0])
        xobs = xobs.replace(to_replace=-9999, value=_np.nan)

        if varname is None:
            return xobs
        else:
            if type(varname) is str:
                data = xobs[varname]
                data.columns = [int(x) for x in data.columns]
                return data
            else:
                cols = [(x[0], int(x[1])) for x in xobs.columns.to_numpy()]
                multi_index = _pd.MultiIndex.from_tuples(cols)
                xobs.columns = multi_index
                return xobs
    else:
        return _pd.DataFrame([])


def merge_timeseries(**kwargs):
    """
    Merge multiple DataFrames into a multi index DataFrame that contains multiple
    time series.

    Example: Create a DataFrame with two variables: precipitation and temperature

    temp = read_hype_timeserie('./Tobs.txt')
    prec = read_hype_timeserie('./Pobs.txt')
    xobs = merge_timeseries(prec=prec, temp=temp)

    :param kwargs:    key and value as parameters
                      value must be a filename or a DataFrame
    :return:          [DataFrame]
    """

    # build multi index
    index_level_1, index_level_2 = [], []

    # Get time series
    xobs = {}
    for key, data in kwargs.items():
        if type(data) is str:
            if data.lower().endswith('.csv'):
                data = _pd.read_csv(data, sep=',', index_col=[0], parse_dates=[0])
            else:
                data = _pd.read_csv(data, sep='\t', index_col=[0], parse_dates=[0])
            data = self._convert_to_number(data)
        if isinstance(data, _pd.DataFrame):
            raise TypeError(f'Wrong data type for < {key} >')

        xobs[key] = data.round(4).fillna(value=-9999)

        index_level_1.extend([key] * xobs[key].shape[1])
        index_level_2.extend(list(xobs[key].columns))

    # Join DataFrames as Multi Index
    multi_index = _pd.MultiIndex.from_tuples(list(zip(index_level_1, index_level_2)),
                                             names=['DATE', '0'])

    xobs_data = _pd.concat(xobs.values(), axis=1)
    xobs_data.columns = multi_index
    xobs_data = xobs_data.sort_index(axis=1, level=0)

    return xobs_data


def melt_timeseries(serie1, serie2, dropna=False):
    """
    Combines two DataFrames of time series into a DataFrame.
    Values from input DataFrames are sorted into two columns (x1, x2),
    and index and columns are set as categories two columns (dates, label).
    This tool requires the same column names in input DataFrames

    :param serie1:       [DataFrame] input time serie 1
    :param serie2:       [DataFrame] input time serie 2
    :param dropna:       [bool] if True, removes rows whit missing values
                           in any column (x1, x2)
    :return:             [DataFrame]
    """

    cols = serie1.columns
    data1, data2, index, labels = [], [], [], []
    for col in cols:
        if col in serie1 and col in serie2:
            data = _pd.concat((serie1.loc[:, col], serie2.loc[:, col]), axis=1)
            if dropna:
                data = data.dropna()
            lab = _np.full(data.shape[0], col, dtype=object)
            data1.extend(data.iloc[:, 0])
            data2.extend(data.iloc[:, 1])
            index.extend(data.index.values)
            labels.extend(lab)

    return _pd.DataFrame({'x1': data1, 'x2': data2, 'label': labels, 'date': index})


# ==============================================================================
# SpaceTime series
# ==============================================================================

def weighted_timeserie(series, basins):
    """
    Weighted sum over rows of multiple time series.
    Weigths are input as a two column DataFrame, where first column
    correspond to the basin id. Second column is the weight, i.e. area

    :param series:       [DataFrame] input multi basin time serie
    :param basins:       [DataFrame] two columns DataFrame. First column is the
                          basin id. Second column is the weight
    :return:             [Series] weighted time serie
    """

    data = series[basins.iloc[:, 0].values]
    weights = basins.set_index(basins.columns[0])
    weights = weights.iloc[:, 0]
    weights /= weights.sum()

    return (data * weights).sum(axis=1)


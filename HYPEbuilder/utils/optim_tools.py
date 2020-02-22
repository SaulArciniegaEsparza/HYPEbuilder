"""
==============================================================================
HYPE hydrological model tools for python

Optimization analysis tools


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

import matplotlib.pyplot as _plt
import seaborn as _sns

from . import time_series as _ts


# ==============================================================================
# Time series for multiple basins
# ==============================================================================

def read_optim_basinserie(folder, subid):
    """
    Read all simulation files of a basin serie
    This tool creates a multi column DataFrame where first level corresponds to
    the simulation number

    :param folder:       [string] existing folder with optimum simulations
    :param subid:        [int] sub basin id
    :return:             [DataFrame]
    """

    dataset = {}

    for fname in _os.listdir(folder):
        if f'{subid:07d}' in fname.lower():
            sumid = int(_os.path.splitext(fname)[0].split('_')[1])
            filename = _os.path.join(folder, fname)
            dataset[sumid] = _ts.read_hype_basinserie(filename)

    if dataset:
        return _ts.merge_timeseries(**dataset)
    else:
        return _pd.DataFrame([])


def read_optim_timeserie(folder, varname, subid=None):
    """
    Read all simulation files of a time serie choosing a variable and basins id
    This tool creates a multi column DataFrame where first level corresponds to
    the simulation number

    :param folder:       [string] existing folder with optimum simulations
    :param varname:      [string] name of variable to read
    :param subid:        [int, tuple, list] basin id. If None, all basins are
                             returned as a DataFrame
    :return:             [DataFrame]
    """

    index_level_1, index_level_2 = [], []
    keys, values = [], []
    varname = varname.lower()

    for fname in _os.listdir(folder):
        if varname in fname.lower():
            sumid = int(_os.path.splitext(fname)[0].split('_')[1])
            filename = _os.path.join(folder, fname)
            values.append(_ts.read_hype_timeserie(filename, subid=subid))
            keys.append(sumid)

    if values:
        for i, key in enumerate(keys):
            index_level_1.extend([key] * values[i].shape[1])
            index_level_2.extend(list(values[i].columns))
        multi_index = _pd.MultiIndex.from_tuples(list(zip(index_level_1, index_level_2)),
                                                 names=['DATE', '0'])
        dataset = _pd.concat(values, axis=1)
        dataset.columns = multi_index
        dataset = dataset.sort_index(axis=1, level=0)

        return dataset
    else:
        return _pd.DataFrame([])


def timeserie_boundaries(data, low='std', middle='mean', upp='std'):
    """
    Returns the lower, middle and upper statistics of a time serie

    :param data:       [DataFrame] input multi timeserie for a single basin
    :param low:        [string] lower limit. Available options are:
                                 'min': minimum value
                                 'std': half standard deviation (default)
                                 'q5' : quantile at 5%
                                 'q10': quantile at 10%
                                 'q25': quantile at 25%
    :param middle:    [string] Middle value. Available options are:
                                'mean'  : average value (default)
                                'median': quantile at 50%
    :param upp:       [string] upper limit. Available options are:
                                 'max': maximum value
                                 'std': half standard deviation (default)
                                 'q95': quantile at 95%
                                 'q90': quantile at 90%
                                 'q75': quantile at 75%
    :return:          [DataFrame] tree columns time serie
    """

    data_stats = _pd.DataFrame(
        _np.zeros((data.shape[0], 3)),
        columns=['low', 'middle', 'upp'],
        index=data.index
    )

    if middle.lower() == 'mean':
        data_stats['middle'] = data.mean(axis=1)
    elif middle.lower() == 'median':
        data_stats['middle'] = data.quantile(0.50, axis=1)

    if low.lower() == 'min':
        data_stats['low'] = data.min(axis=1)
    elif low.lower() == 'std':
        data_stats['low'] = data_stats['middle'] - data.std(axis=1) / 2.0
    elif low.lower() == 'q5':
        data_stats['low'] = data.quantile(0.05, axis=1)
    elif low.lower() == 'q10':
        data_stats['low'] = data.quantile(0.10, axis=1)
    elif low.lower() == 'q25':
        data_stats['low'] = data.quantile(0.25, axis=1)

    if upp.lower() == 'max':
        data_stats['upp'] = data.max(axis=1)
    elif upp.lower() == 'std':
        data_stats['upp'] = data_stats['middle'] + data.std(axis=1) / 2.0
    elif upp.lower() == 'q95':
        data_stats['upp'] = data.quantile(0.95, axis=1)
    elif upp.lower() == 'q90':
        data_stats['upp'] = data.quantile(0.90, axis=1)
    elif upp.lower() == 'q75':
        data_stats['upp'] = data.quantile(0.75, axis=1)

    return data_stats


# ==============================================================================
# Optimum parameters
# ==============================================================================

def read_subass(folder, simid=1):
    """
    Reads all the subass files in a folder as a DataFrame
    Column "Simulation" corresponds to the simulation number

    :param folder:    [string] folder name
    :param simid:     [int] criteria number used for optimization. Default 1
    :return:          [DataFrame] subass data for all simulations
    """

    subass_data = []
    for fname in _os.listdir(folder):
        if f'subass{simid}' in fname.lower():
            filename = _os.path.join(folder, fname)
            sumid = int(_os.path.splitext(fname)[0].split('_')[1])
            data = _pd.read_csv(filename, sep='\t', skiprows=1)
            data['Simulation'] = sumid
            subass_data.append(data)
    if subass_data:
        subass = _pd.concat(subass_data, axis=0)
        subass.set_index('Simulation', inplace=True)
        subass.reset_index(inplace=True)
        return subass
    else:
        return _pd.DataFrame([])


def read_bestsims(folder):
    """
    Reads a bestsims file as a DataFrame

    :param folder:     [string] folder where bestsims.txt is located
    :return:           [DataFrame]
    """

    filename = _os.path.join(folder, 'bestsims.txt')
    if _os.path.exists(filename):
        return _pd.read_csv(filename, sep=',')
    else:
        return _pd.DataFrame([])


# ==============================================================================
# Plots
# ==============================================================================

def scatter_bestsims(bestsims, metric='rr2', variables=None, col_wrap=4,
                     show_best=True, **kwargs):
    """
    Scatterplots between an error metric and optimized variables

    :param bestsims:       [DataFrame] input bestsims data
    :param metric:         [string] name of the error metric
    :param variables:      [tuple, list, array] name of variables to plot.
                            If None, all variables are plotted
    :param col_wrap:       [int] number of columns for grid of axis
    :param show_best:      [bool] if True, first row of parameters is plot as red points
    :param kwargs:         optional arguments as color and marker points
    :return:               Seaborn FaceGrid object
    """

    # Default metric names
    metric_names = {
        'rr2': 'Regional NSE',
        'sr2': 'Spatial NSE',
        'mr2': 'Average NSE',
        'rmae': 'Regional MAE',
        'sre': 'Spatial RE',
        'rre': 'Regional RE',
        'mre': 'Average RE',
        'rra': 'Regional RA',
        'sra': 'Spatial RA',
        'mra': 'Median RA',
        'tau': 'Kendalls Tau',
        'md2': 'Median NSE',
        'mda': 'Median RA',
        'mrs': 'Average RSDE',
        'mcc': 'Average CC',
        'mdkg': 'Median KGE',
        'akg': 'Average KGE',
        'mar': 'Average ARE',
        'mdnr': 'Median NRMSE',
        'mnw': 'Mean NSEW'
    }

    metric = metric.lower()
    if metric not in metric_names:
        raise ValueError(f'metric {metric} is not recognized.')
    metric_name = metric_names[metric]

    # Get variable names
    var_names = [x.split('.')[0] for x in list(bestsims.columns[24:])]
    var_values = bestsims.iloc[:, 24:]
    var_values.columns = var_names

    # Check names in columns
    if variables is None:
        variables = list(_np.unique(var_names))
    else:
        for vname in variables:
            if vname not in var_names:
                raise ValueError(f'Variable < {vname} > is not in bestsims')

    var_values = var_values.loc[:, variables]

    # Melt data
    data = _pd.concat((bestsims[[metric]], var_values), axis=1)
    data = data.melt(id_vars=[metric])
    data.columns = [metric_name, 'variable', 'value']

    # Create FaceGrid
    color = kwargs.get('color', 'grey')
    marker = kwargs.get('marker', '.')
    grid = _sns.FacetGrid(data, col="variable", palette="tab20c",
                         col_wrap=col_wrap, sharex=False, despine=False)
    grid.map(_plt.scatter, 'value', metric_name, marker=marker, color=color)

    axes = grid.axes
    for ax in axes:
        title = ax.get_title()
        title = title.replace('variable', '').replace('=', '').replace(" ", "")
        ax.set_title(title)
        ax.set_xlabel('')
        if show_best:
            x = var_values.iloc[0, :].loc[title]
            if type(x) in (int, float, _np.float16, _np.float32, _np.float64):
                x = [x]
            y = [bestsims.iloc[0, :].loc[metric]] * len(x)
            ax.plot(x, y, 'r.')

    grid.fig.tight_layout()
    return grid


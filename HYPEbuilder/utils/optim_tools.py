"""
==============================================================================
HYPE hydrological model tools for python

Opyimization analyzis tools


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

    varname = varname.lower()
    dataset = {}

    for fname in _os.listdir(folder):
        if varname in fname.lower():
            sumid = int(_os.path.splitext(fname)[0].split('_')[1])
            filename = _os.path.join(folder, fname)
            dataset[sumid] = _ts.read_hype_timeserie(filename, subid=subid)

    if dataset:
        return _ts.merge_timeseries(**dataset)
    else:
        return _pd.DataFrame([])


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

    :param filename:     [string] folder where bestsims.txt is located
    :return:             [DataFrame]
    """

    filename = _os.path.join(folder, 'bestsims.txt')
    if _os.path.exists(filename):
        return _pd.read_csv(filename, sep=',')
    else:
        return _pd.DataFrame([])





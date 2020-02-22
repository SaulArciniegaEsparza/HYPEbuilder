"""
==============================================================================
HYPE hydrological model tools for python

Graphic tools for results analysis and data inspection


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
======
"""

import numpy as _np
import pandas as _pd

import matplotlib.pyplot as _plt
import seaborn as _sns

from .optim_tools import timeserie_boundaries as _tsb


# ==============================================================================
# Time series analysis
# ==============================================================================

def plot_boundaries(data, color='k', alpha=0.15, ax=None, **kwargs):
    """
    Plot a time serie threshold using several series. Threshold is computed
    as the standard deviation.
    Input data can be a util.optim_tools.timeserie_boundaries result

    :param data:      [DataFrame] input time series
    :param color:     [string] color for area and line
    :param alpha:     [float] opacity for area plot. 0 < alpha < 1
    :param ax:        [axes] matplotlib axes object to plot. Optional
    :param kwargs:    aditional arguments as figsize, ylabel
    :return:          axes plot, or if input ax is None, returns fig and axes
    """

    op = False
    if ax is None:
        op = True
        figsize = kwargs.get('figsize', None)
        fig, ax = _plt.subplots(figsize=figsize)

    if 'low' in data and 'middle' in data and 'upp' in data:
        data_stats = data.loc[:, ['low', 'middle', 'upp']]
    else:
        data_stats = _tsb(data, low='std', middle='mean', upp='std')

    # Plot data
    ax.fill_between(data_stats.index, y1=data_stats.loc[:, 'low'],
                    y2=data_stats.loc[:, 'upp'], color=color,
                    alpha=alpha, label='threshold')
    ax.plot(data_stats.index, data_stats.loc[:, 'middle'], color=color)

    # Axis properties
    ylabel = kwargs.get('ylabel', 'Value')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('', fontsize=12)

    if op:
        fig.tight_layout()
        return fig, ax
    else:
        return ax


# ==============================================================================
# Flow (streamflow, baseflow) plots
# ==============================================================================

def flow_duration_curve(data, log_scale=True):
    pass


# ==============================================================================
# GeoData and Geoclass
# ==============================================================================

def plot_watershed_slc():
    pass





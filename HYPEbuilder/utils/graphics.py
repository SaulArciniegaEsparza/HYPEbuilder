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
from matplotlib.collections import LineCollection as _LineCollection
from matplotlib.gridspec import GridSpec as _GridSpec

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
# Water balance
# ==============================================================================

def plot_water_balance(data_bl, data_tl, data_br=None, data_tr=None,
                       **kwargs):
    """
    Plot multiple time series in two axes. Top axes is half size than bottom axes.
    Each axes can be used to plot a double y axis plot

    Input data is a dictionary with the following keys and values:
        'data':       [Serie, DataFrame] data to be plotted
        'ylabel':     [string] ylabel for the current axis
        'kind':       [string] (Optional) kind of plot ('line', 'bar')
        'color':      [string, tuple, list] (optional) color or colors
        'linestyle':  [string, tuple, list] (optional) line style (default:'-')


    :param data_bl:      [dict] input data and attributes for bottom left axis
    :param data_tl:      [dict] input data and attributes for top left axis
    :param data_br:      [dict] input data and attributes for bottom right axis
    :param data_tr:      [dict] input data and attributes for top right axis
    :param kwargs:       additional plot arguments 'figsize', 'fontsize', 'title',
                            'rotation'
    :return:
    """

    # Default parameters
    figsize = kwargs.get('figsize', None)
    fontsize = kwargs.get('fontsize', 12)
    title = kwargs.get('title', None)
    rotation = kwargs.get('rotation', 0)

    # Create fig and axis
    fig = _plt.figure(figsize=figsize)
    gs = _GridSpec(3, 1, figure=fig)
    ax_bl = fig.add_subplot(gs[1:, 0])
    ax_tl = fig.add_subplot(gs[0, 0], sharex=ax_bl)
    axes = [[ax_bl], [ax_tl]]

    # plot bottom left
    kind = data_bl.get('kind', 'line')
    color = data_bl.get('color', None)
    linestyle = data_bl.get('linestyle', '-')
    ylabel = data_bl.get('ylabel', '')
    data = data_bl['data']
    data.plot(color=color, kind=kind, linestyle=linestyle, ax=ax_bl)
    ax_bl.set_ylabel(ylabel, fontsize=fontsize)
    ax_bl.set_xlabel('')

    # plot upper right
    if data_br is not None:
        ax_br = ax_bl.twinx()
        axes[0].append(ax_br)
        kind = data_br.get('kind', 'line')
        color = data_br.get('color', None)
        linestyle = data_br.get('linestyle', '-')
        ylabel = data_br.get('ylabel', '')
        data = data_br['data']
        data.plot(color=color, kind=kind, linestyle=linestyle, ax=ax_br)
        ax_br.set_ylabel(ylabel, fontsize=fontsize)
        ax_br.set_xlabel('')

    # plot top left
    kind = data_tl.get('kind', 'line')
    color = data_tl.get('color', None)
    linestyle = data_tl.get('linestyle', '-')
    ylabel = data_tl.get('ylabel', '')
    data = data_tl['data']
    data.plot(color=color, kind=kind, linestyle=linestyle, ax=ax_tl)
    ax_tl.set_ylabel(ylabel, fontsize=fontsize)
    ax_tl.set_xlabel('')

    if data_tr is not None:
        ax_tr = ax_tl.twinx()
        axes[1].append(ax_tr)
        kind = data_tr.get('kind', 'line')
        color = data_tr.get('color', None)
        linestyle = data_tr.get('linestyle', '-')
        ylabel = data_tr.get('ylabel', '')
        data = data_tr['data']
        data.plot(color=color, kind=kind, linestyle=linestyle, ax=ax_tr)
        ax_tr.set_ylabel(ylabel, fontsize=fontsize)
        ax_tr.set_xlabel('')

    # Set axis attributes
    if rotation > 0:
        ax_bl.tick_params(axis='x', rotation=rotation)
    fig.suptitle(title, fontsize=fontsize)
    fig.tight_layout()
    return axes


# ==============================================================================
# Data Analysis
# ==============================================================================

def double_yaxis(data_left, data_right, ax=None, yl_left=None, yl_right=None,
                 c_left=None, c_right=None, ls_left=None, ls_right=None, **kwargs):
    """
    Create a two y axis with multiple lines

    :param data_left:       [Serie, DataFrame] input data for right axis
    :param data_right:      [Serie, DataFrame] input data for left axis
    :param ax:              [axes] (optional) input matplotlib axes object
    :param yl_left:         [string] ylabel left axis
    :param yl_right:        [string] ylabel right axis
    :param c_left:          [string, tuple, list] line color or colors for left axis
    :param c_right:         [string, tuple, list] line color or colors for right axis
    :param ls_left:         [string] line style for lines on left axis
    :param ls_right:        [string] line style for lines on right axis
    :param kwargs:          additional arguments: 'figsize', 'fontsize', 'rotation'
    :return:
    """

    # Default parameters
    if yl_left is None:
        yl_left = ''
    if yl_right is None:
        yl_right = ''
    if ls_left is None:
        ls_left = '-'
    if ls_right is None:
        ls_right = '-'
    figsize = kwargs.get('figsize', None)
    fontsize = kwargs.get('fontsize', 12)
    rotation = kwargs.get('rotation', 0)

    op = False
    if ax is None:
        op = True
        fig, axl = _plt.subplots(figsize=figsize)
    else:
        axl = ax
    axr = axl.twinx()

    # Plot left axis
    data_left.plot(color=c_left, linestyle=ls_left, ax=axl)
    axl.set_ylabel(yl_left, fontsize=fontsize)
    axl.set_xlabel('')

    # Plot right axis
    data_right.plot(color=c_right, linestyle=ls_right, ax=axr)
    axr.set_ylabel(yl_right, fontsize=fontsize)
    axr.set_xlabel('')

    # Axis attributes
    if rotation > 0:
        axl.tick_params(axis='x', rotation=rotation)
    if op:
        fig.tight_layout()
        return fig, ax
    else:
        return ax


def multiple_lines(data, attr, ax=None, **kwargs):
    """
    Plot multiple lines using an attribute value as color

    :param data:          [DataFrames] input time series
    :param attr:          [Serie] input attribute
    :param ax:            [axis] optional input axis
    :param kwargs:        additional arguments:
                            'xlabel'   >  [string] input xlabel
                            'ylabel'   >  [string] input ylabel
                            'figsize'  >  [tuple] input fig size
                            'fontsize' >  [int] fontsize value
                            'cmap'     >  [stirng] cmap name
                            'label'    >  [string] colorbar label
                            'yscale'   >  [string] y axis scale 'linear' or 'log'
    :return:             axes plot, or if input ax is None, returns fig and axes
    """

    # Default parameters
    xlabel = kwargs.get('xlabel', '')
    ylabel = kwargs.get('ylabel', '')
    yscale = kwargs.get('yscale', 'linear')
    figsize = kwargs.get('figsize', None)
    fontsize = kwargs.get('fontsize', 12)
    cmap = kwargs.get('cmap', 'viridis')
    label = kwargs.get('label', None)

    # Associate axes
    op = False
    if ax is None:
        op = True
        fig, ax = _plt.subplots(figsize=figsize)
    else:
        fig = _plt.gcf()

    # Create multi lines
    lc = _multiline(
        data.index.values,
        data.values,
        attr.loc[data.index.values].values,
        cmap=cmap,
        lw=1
    )

    # Set colorbar
    axcb = fig.colorbar(lc)

    # Set axis attributes
    if label:
        axcb.set_label(label, fontsize=fontsize)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid(True, which='both')

    if op:
        fig.tight_layout()
        return fig, ax
    else:
        return ax


def _multilines(xs, ys, c, ax, **kwargs):
    """
    Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # create LineCollection
    segments = [_np.column_stack([xs, ys[:, i]]) for i in range(ys.shape[1])]
    lc = _LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(_np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


# ==============================================================================
# GeoData and Geoclass
# ==============================================================================

def plot_watershed_slc():
    pass





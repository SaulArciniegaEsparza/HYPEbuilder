"""
==============================================================================
HYPE hydrological model tools for python

Metrics for model evaluation


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
======
"""

import numpy as _np
import pandas as _pd


# ==============================================================================
# Metrics
# ==============================================================================

def nse(obs, sim):
    """
    Computes the Nash-Sutcliffe Efficiency score over a single time serie or
    on multiple time series

    :param obs:     [Serie, DataFrame] observed values
    :param sim:     [Serie, DataFrame] simulated values
    :return:        [float, Serie] NSE score or scores values
    """

    if isinstance(obs, _pd.DataFrame) and isinstance(sim, _pd.DataFrame):
        keys = [key for key in obs.columns if key in sim.columns]
        n = len(keys)
        if n > 0:
            scores = _pd.Series(_np.zeros(n, dtype=_np.float32), index=keys)
            for key in keys:
                scores.loc[key] = _nash_sutcliffe_efficiency(obs[key], sim[key])
            return scores
        else:
            return _pd.Series([])

    elif isinstance(obs, _pd.Series) and isinstance(sim, _pd.Series):
        return _nash_sutcliffe_efficiency(obs, sim)

    else:
        raise TypeError('Bad input values. Arguments obs and sim must be Series or DataFrames!')


def nse_log(obs, sim):
    """
    Computes the log of Nash-Sutcliffe Efficiency score over a single time serie or
    on multiple time series

    :param obs:     [Serie, DataFrame] observed values
    :param sim:     [Serie, DataFrame] simulated values
    :return:        [float, Serie] NSElog score or scores values
    """

    if isinstance(obs, _pd.DataFrame) and isinstance(sim, _pd.DataFrame):
        keys = [key for key in obs.columns if key in sim.columns]
        n = len(keys)
        if n > 0:
            scores = _pd.Series(_np.zeros(n, dtype=_np.float32), index=keys)
            for key in keys:
                scores.loc[key] = _log_nash_sutcliffe_efficiency(obs[key], sim[key])
            return scores
        else:
            return _pd.Series([])

    elif isinstance(obs, _pd.Series) and isinstance(sim, _pd.Series):
        return _log_nash_sutcliffe_efficiency(obs, sim)

    else:
        raise TypeError('Bad input values. Arguments obs and sim must be Series or DataFrames!')


def kge(obs, sim):
    """
    Computes the Kling-Gupta Efficiency score over a single time serie or
    on multiple time series

    :param obs:     [Serie, DataFrame] observed values
    :param sim:     [Serie, DataFrame] simulated values
    :return:        [float, Serie] KGE score or scores values
    """

    if isinstance(obs, _pd.DataFrame) and isinstance(sim, _pd.DataFrame):
        keys = [key for key in obs.columns if key in sim.columns]
        n = len(keys)
        if n > 0:
            scores = _pd.Series(_np.zeros(n, dtype=_np.float32), index=keys)
            for key in keys:
                scores.loc[key] = _kling_gupta_efficiency(obs[key], sim[key])
            return scores
        else:
            return _pd.Series([])

    elif isinstance(obs, _pd.Series) and isinstance(sim, _pd.Series):
        return _kling_gupta_efficiency(obs, sim)

    else:
        raise TypeError('Bad input values. Arguments obs and sim must be Series or DataFrames!')


def bias(obs, sim):
    """
    Computes the Bias score over a single time serie or
    on multiple time series

    :param obs:     [Serie, DataFrame] observed values
    :param sim:     [Serie, DataFrame] simulated values
    :return:        [float, Serie] Bias score or scores values
    """

    if isinstance(obs, _pd.DataFrame) and isinstance(sim, _pd.DataFrame):
        keys = [key for key in obs.columns if key in sim.columns]
        n = len(keys)
        if n > 0:
            scores = _pd.Series(_np.zeros(n, dtype=_np.float32), index=keys)
            for key in keys:
                scores.loc[key] = _bias(obs[key], sim[key])
            return scores
        else:
            return _pd.Series([])

    elif isinstance(obs, _pd.Series) and isinstance(sim, _pd.Series):
        return _bias(obs, sim)

    else:
        raise TypeError('Bad input values. Arguments obs and sim must be Series or DataFrames!')


def bias_relative(obs, sim):
    """
    Computes the Relative Bias score over a single time serie or
    on multiple time series

    :param obs:     [Serie, DataFrame] observed values
    :param sim:     [Serie, DataFrame] simulated values
    :return:        [float, Serie] RB score or scores values
    """

    if isinstance(obs, _pd.DataFrame) and isinstance(sim, _pd.DataFrame):
        keys = [key for key in obs.columns if key in sim.columns]
        n = len(keys)
        if n > 0:
            scores = _pd.Series(_np.zeros(n, dtype=_np.float32), index=keys)
            for key in keys:
                scores.loc[key] = _relative_bias(obs[key], sim[key])
            return scores
        else:
            return _pd.Series([])

    elif isinstance(obs, _pd.Series) and isinstance(sim, _pd.Series):
        return _relative_bias(obs, sim)

    else:
        raise TypeError('Bad input values. Arguments obs and sim must be Series or DataFrames!')


def rmse(obs, sim):
    """
    Computes the Root Mean Square Error over a single time serie or
    on multiple time series

    :param obs:     [Serie, DataFrame] observed values
    :param sim:     [Serie, DataFrame] simulated values
    :return:        [float, Serie] RMSE score or scores values
    """

    if isinstance(obs, _pd.DataFrame) and isinstance(sim, _pd.DataFrame):
        keys = [key for key in obs.columns if key in sim.columns]
        n = len(keys)
        if n > 0:
            scores = _pd.Series(_np.zeros(n, dtype=_np.float32), index=keys)
            for key in keys:
                scores.loc[key] = _root_mean_square_error(obs[key], sim[key])
            return scores
        else:
            return _pd.Series([])

    elif isinstance(obs, _pd.Series) and isinstance(sim, _pd.Series):
        return _root_mean_square_error(obs, sim)

    else:
        raise TypeError('Bad input values. Arguments obs and sim must be Series or DataFrames!')


def mae(obs, sim):
    """
    Computes the Mean Absolute Error over a single time serie or
    on multiple time series

    :param obs:     [Serie, DataFrame] observed values
    :param sim:     [Serie, DataFrame] simulated values
    :return:        [float, Serie] MAE score or scores values
    """

    if isinstance(obs, _pd.DataFrame) and isinstance(sim, _pd.DataFrame):
        keys = [key for key in obs.columns if key in sim.columns]
        n = len(keys)
        if n > 0:
            scores = _pd.Series(_np.zeros(n, dtype=_np.float32), index=keys)
            for key in keys:
                scores.loc[key] = _mean_absolute_error(obs[key], sim[key])
            return scores
        else:
            return _pd.Series([])

    elif isinstance(obs, _pd.Series) and isinstance(sim, _pd.Series):
        return _mean_absolute_error(obs, sim)

    else:
        raise TypeError('Bad input values. Arguments obs and sim must be Series or DataFrames!')


def pearson_correlation(obs, sim):
    """
    Computes the Pearson Correlation Coefficient over a single time serie or
    on multiple time series

    :param obs:     [Serie, DataFrame] observed values
    :param sim:     [Serie, DataFrame] simulated values
    :return:        [float, Serie] r score or scores values
    """

    if isinstance(obs, _pd.DataFrame) and isinstance(sim, _pd.DataFrame):
        keys = [key for key in obs.columns if key in sim.columns]
        n = len(keys)
        if n > 0:
            scores = _pd.Series(_np.zeros(n, dtype=_np.float32), index=keys)
            for key in keys:
                scores.loc[key] = _pearson_correlation(obs[key], sim[key])
            return scores
        else:
            return _pd.Series([])

    elif isinstance(obs, _pd.Series) and isinstance(sim, _pd.Series):
        return _pearson_correlation(obs, sim)

    else:
        raise TypeError('Bad input values. Arguments obs and sim must be Series or DataFrames!')


# ==============================================================================
# Core metrics
# ==============================================================================

def _common_period(serie1, serie2):
    """
    Create a DataFrame with common non null data

    :param serie1:    [Serie] input serie with datetime as index
    :param serie2:    [Serie] input serie with datetime as index
    :return:          [DataFrame] output DataFrame with two columns
    """

    data = _pd.concat((serie1, serie2), axis=1)
    data.columns = ['serie1', 'serie2']
    data.dropna(inplace=True)
    return data


def _nash_sutcliffe_efficiency(obs, sim):
    """
    Compute Nash-Sutcliffe Efficiency score
    NaNs are ignored

    :param obs:    [Serie] observed values
    :param sim:    [Serie] simulated values
    :return:       [float] SNE score
    """

    data = _common_period(obs, sim)
    rm = data.iloc[:, 0].mean()
    part1 = ((data.iloc[:, 1] - data.iloc[:, 0]) ** 2).sum()
    part2 = ((data.iloc[:, 0] - rm) ** 2).sum()

    return 1.0 - part1 / part2


def _log_nash_sutcliffe_efficiency(obs, sim):
    """
    Compute log Nash-Sutcliffe Efficiency score
    NaNs and zeros are ignored

    :param obs:    [Serie] observed values
    :param sim:    [Serie] simulated values
    :return:       [float] SNElog score
    """

    data = _common_period(obs, sim)
    data = _np.log(data.loc[(data.iloc[:, 0] > 0) & (data.iloc[:, 1] > 0), :])
    rm = data.iloc[:, 0].mean()
    part1 = ((data.iloc[:, 1] - data.iloc[:, 0]) ** 2).sum()
    part2 = ((data.iloc[:, 0] - rm) ** 2).sum()

    return 1.0 - part1 / part2


def _kling_gupta_efficiency(obs, sim):
    """
    Compute Kling-Gupta efficiency
    NaNs are ignored

    :param obs:    [Serie] observed values
    :param sim:    [Serie] simulated values
    :return:       [float] KGE score
    """

    data = _common_period(obs, sim)
    mean = data.mean().values
    std = data.std().values
    cc = data.corr().iloc[0, 1]
    part1 = (cc - 1.0) ** 2.0
    part2 = (std[1] / std[0] - 1.0) ** 2.0
    part3 = (mean[1] / mean[0] - 1.0) ** 2.0
    return 1 - (part1 + part2 + part3) ** 0.5


def _bias(obs, sim):
    """
    Computes the bias between observed and simulated
    NaNs are ignored

    :param obs:    [Serie] observed values
    :param sim:    [Serie] simulated values
    :return:       [float] bias value
    """

    data = _common_period(obs, sim)
    n = data.shape[0]

    return (data.iloc[:, 1] - data.iloc[:, 0]).sum() / n


def _relative_bias(obs, sim):
    """
    Computes the relative bias between observed and simulated
    NaNs are ignored

    :param obs:    [Serie] observed values
    :param sim:    [Serie] simulated values
    :return:       [float] relative bias value
    """

    data = _common_period(obs, sim)
    part1 = (data.iloc[:, 1] - data.iloc[:, 0]).sum()
    part2 = data.iloc[:, 0].sum()

    return part1 / part2


def _root_mean_square_error(obs, sim):
    """
    Computes the root mean square error
    NaNs are ignored

    :param obs:     [Serie] observed values
    :param sim:     [Serie] simulated values
    :return:        [float] relative bias value
    """

    data = _common_period(obs, sim)
    n = data.shape[0]

    return (((data.iloc[:, 1] - data.iloc[:, 0]) ** 2.0).sum() / n) ** 0.5


def _mean_absolute_error(obs, sim):
    """
    Computes the mean absolute error
    NaNs are ignored

    :param obs:     [Serie] observed values
    :param sim:     [Serie] simulated values
    :return:        [float] mean absolute error
    """

    data = _common_period(obs, sim)
    n = data.shape[0]

    return (data.iloc[:, 1] - data.iloc[:, 0]).abs().sum() / n


def _pearson_correlation(obs, sim):
    """
    Computes the mean absolute error
    NaNs are ignored

    :param obs:     [Serie] observed values
    :param sim:     [Serie] simulated values
    :return:        [float] mean absolute error
    """

    data = _common_period(obs, sim)
    return data.corr().iloc[0, 1]


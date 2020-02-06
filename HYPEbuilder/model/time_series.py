"""
==============================================================================
HYPE hydrological model tools for python

Time Forcings and Results series tools


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
==============================================================================
"""

import os
import numpy as np
import pandas as pd


# ==============================================================================
# Define constants
# ==============================================================================

INPUT_OBS = {
    'pobs': 'Pobs',
    'tobs': 'Tobs',
    'qobs': 'Qobs',
    'rhobs': 'RHobs',
    'sfobs': 'SFobs',
    'swobs': 'SWobs',
    'tminobs': 'TMINobs',
    'tmaxobs': 'TMAXobs',
    'uobs': 'Uobs',
}

CLASS1 = type(pd.Series())
CLASS2 = type(pd.DataFrame())


# ==============================================================================
# Observation class
# ==============================================================================

class Forcings(object):

    def __init__(self, path):
        self.path = os.path.join(path, 'forcings')

    def __repr__(self):
        return 'HYPEbuilder.Forcings'

    @staticmethod
    def _convert_to_number(data):
        """
        Converts columns from DataFrame to numeric format
        """
        data1 = data.copy()
        for col in data.columns:
            data1.loc[:, col] = pd.to_numeric(data.loc[:, col], errors='coerce')
        return data1

    def check_if_path_exist(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def import_obs(self, **kwargs):
        """Import forcing observations to current model"""

        self.check_if_path_exist()
        for key, data in kwargs.items():
            key = key.lower()
            if key in INPUT_OBS:
                saveas = os.path.join(self.path, INPUT_OBS[key] + '.txt')
                if type(data) is str:
                    if data.lower().endswith('.csv'):
                        data = pd.read_csv(data, sep=',', index_col=[0], parse_dates=[0])
                    else:
                        data = pd.read_csv(data, sep='\t', index_col=[0], parse_dates=[0])
                if type(data) is not CLASS2:
                    raise TypeError(f'Wrong data type for < {key} >')

                data = self._convert_to_number(data)
                data.index.name = 'DATE'
                data = data.round(4).fillna(value=-9999)
                data.to_csv(saveas, sep='\t')

    def import_xobs(self, **kwargs):
        """Imports Xobservations to the model"""

        # build multi index
        index_level_1, index_level_2 = [], []

        # Get time series
        xobs = {}
        for key, data in kwargs.items():
            if type(data) is str:
                if data.lower().endswith('.csv'):
                    data = pd.read_csv(data, sep=',', index_col=[0], parse_dates=[0])
                else:
                    data = pd.read_csv(data, sep='\t', index_col=[0], parse_dates=[0])
                data = self._convert_to_number(data)
            if type(data) is not CLASS2:
                raise TypeError(f'Wrong data type for < {key} >')

            xobs[key] = data.round(4).fillna(value=-9999)

            index_level_1.extend([key] * xobs[key].shape[1])
            index_level_2.extend(list(xobs[key].columns))

        # Join DataFrames as Multi Index
        multi_index = pd.MultiIndex.from_tuples(list(zip(index_level_1, index_level_2)),
                                                names=['DATE', '0'])

        xobs_data = pd.concat(xobs.values(), axis=1)
        xobs_data.columns = multi_index
        xobs_data = xobs_data.sort_index(axis=1, level=0)

        # Write file using first line as comment
        filename = os.path.join(self.path, 'Xobs.txt')
        self.check_if_path_exist()

        with open(filename, 'w') as fout:
            fout.write('! Xobservations\n')
        xobs_data.to_csv(filename, sep='\t', index_label=False, mode='a')

    def import_xobs_from_multi_index(self, xobs):
        """Imports Xobservations to the model from a multi index DataFrame"""

        filename = os.path.join(self.path, 'Xobs.txt')
        self.check_if_path_exist()
        if isinstance(xobs.columns, pd.MultiIndex):
            with open(filename, 'w') as fout:
                fout.write('! Xobservations\n')
            xobs_data = xobs_data.fillna(value=-9999)
            xobs_data.to_csv(filename, sep='\t', index_label=False, mode='a')
        else:
            raise TypeError('Input DataFrame must contain Multi Index columns!')

    def get_obs(self, varname='pobs', subid=None):
        """Export forcings observations to DataFrames using catchments subids"""

        varname = varname.lower()
        if varname in INPUT_OBS:
            filename = os.path.join(self.path, f'{INPUT_OBS[varname]}.txt')
            if os.path.exists(filename):
                if subid is not None:
                    if type(subid) in (int, float):
                        subids = ['DATE'] + [str(int(subid))]
                    elif type(subid) in (tuple, list, np.ndarray):
                        subids = ['DATE'] + [str(int(x)) for x in subid]
                    else:
                        subids = None
                else:
                    subids = None
                data = pd.read_csv(filename, sep='\t', index_col=[0], parse_dates=[0],
                                   usecols=subids)
                data = data.replace(to_replace=-9999, value=np.nan)
                data.columns = [int(x) for x in data.columns]
                return data
            else:
                return pd.DataFrame([])
        else:
            raise ValueError(f'Variable < {varname} > is not a forcing observation!')

    def get_xobs(self, varname=None):
        """Returns the xobs as MultiIndex DataFrame or if varname returns a DataFrame"""
        filename = os.path.join(self.path, 'Xobs.txt')
        if os.path.exists(filename):
            xobs = pd.read_csv(filename, sep='\t', skiprows=1, header=[0, 1], index_col=[0],
                               parse_dates=[0])
            xobs = xobs.replace(to_replace=-9999, value=np.nan)

            if varname is None:
                return xobs
            else:
                if type(varname) is str:
                    return xobs[varname]
                else:
                    return xobs


# ==============================================================================
# Time Series Results class
# ==============================================================================

class Results(object):

    def __init__(self, path):
        self.path = os.path.join(path, 'results')

    def __repr__(self):
        return 'hypeBUILDER.Results'

    def check_if_path_exist(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def get_time_series(self, variable, subid=None):

        self.check_if_path_exist()

        if subid is not None:
            if type(subid) in (int, float):
                subid = ['DATE'] + [str(int(subid))]
            elif type(subid) in (tuple, list, np.ndarray):
                subid = ['DATE'] + [str(int(x)) for x in subid]
            else:
                subid = None
        else:
            subid = None

        filename = os.path.join(self.path, f'time{str(variable).upper()}.txt')
        if os.path.exists(filename):
            data = pd.read_csv(filename, sep='\t', skiprows=1, index_col=[0],
                               parse_dates=[0], usecols=subid)
            data = data.replace(to_replace=-9999, value=np.nan)
            data.index.name = 'DATE'
            data.columns = [int(x) for x in data.columns]
            return data
        else:
            return pd.DataFrame([])

    def get_basin_series(self, subid):

        self.check_if_path_exist()

        filename = os.path.join(self.path, f'{int(subid):07d}.txt')
        if os.path.exists(filename):
            data = pd.read_csv(filename, sep='\t', skiprows=range(1, 2), index_col=[0],
                               parse_dates=[0])
            data = data.replace(to_replace=-9999, value=np.nan)
            data.index.name = 'DATE'
            return data
        else:
            return pd.DataFrame([])

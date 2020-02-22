"""
==============================================================================
HYPE hydrological model tools for python

Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
==============================================================================
"""

# Import modules
import os as _os
import numpy as _np
import pandas as _pd
import geopandas as _gpd

import numba as _nb

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


# Get types of different objects
CLASS1 = type(_pd.DataFrame())
CLASS2 = type(_gpd.GeoDataFrame())


# ==============================================================================
# GeoDataClass
# ==============================================================================

class GeoData:
    def __init__(self, geodata):
        """
        GeoData class allows to user work with basins properties and find
        downstream and upstream basins connections following the flow direction.


        INPUTS:
            geodata    [string, DataFrame, GeoDataFrame] input subbassins
                         table, shape file, DataFrame of GeoDataFrame
        """

        if type(geodata) is str:
            if not _os.path.exists(geodata):
                raise IOError(f'GeoData file < {geodata} > does not exist!')

            if geodata.endswith('.csv'):
                geodata = _pd.read_csv(geodata)
            elif geodata.endswith('.txt'):
                geodata = _pd.read_csv(geodata, sep='\t')
            else:
                geodata = _gpd.read_file(geodata)

        if type(geodata) is CLASS1:
            pass
        elif type(geodata) is CLASS2:
            geodata = _pd.DataFrame(geodata.drop('geometry', axis=1))
        else:
            raise TypeError(('geodata input must be a vector/table file '
                             'or a DataFrame/GeoDataFrame object!'))

        # Convert columns to integers
        key_int = ['subid', 'maindown', 'region', 'parreg', 'petmodel']
        for key in key_int:
            geodata.loc[:, key] = geodata.loc[:, key].astype(int)

        # Round float data
        dtypes = geodata.dtypes
        for i in range(geodata.shape[1]):
            if dtypes[i] in (float, _np.float16, _np.float32, _np.float64):
                geodata.iloc[:, i] = geodata.iloc[:, i].round(4)

        self.data = geodata  # save data

    def __repr__(self):
        return 'HYPEbuilder.GeoData'

    def __str__(self):
        return str(self.data)

    def info(self):
        t = "\n\nHYPEbuilder GeoData\n"
        t += " Number of basins: {}\n".format(self.data.shape[0])
        t += " Total area (km2): {:.3f}\n".format(self.data['area'].sum() / 1e6)
        t += "No. Header basins: {}\n".format(len(self.find_header_basins(True)))
        t += "No. Outlet basins: {}\n".format(len(self.find_outlet_basins(True)))
        return t

    def find_header_basins(self, only_subid=False):
        """
        Returns the subid of header basins.
        Header basins do not have upstream basins.

        INPUTS:
            only_subid    [bool] if True, only an array of basins subids is
                           returned. If False (default), a DataFrame is returned.

        """

        # Extract subid and maindown as arrays
        subid = self.data['subid'].values
        maindown = self.data['maindown'].values
        # Compute downstream connectivity matrix
        connect = downstream_id(subid, maindown)
        num_basins = connect.sum(axis=0)
        mask = num_basins == 0  # find header basins
        if only_subid:
            outdata = subid[mask]
        else:
            outdata = self.data.loc[mask, :]
        return outdata

    def find_outlet_basins(self, only_subid=False):
        """
        Returns the subid of outlet basins from a DataFrame/GeoDataFrame
        that contains the subid and maindown codes.
        Outlet basins do not have downstream basins.


        NOTE: subid and maindown inputs must be defined at the same time
        """

        # Extract subid and maindown as arrays
        subid = self.data['subid'].values
        maindown = self.data['maindown'].values
        # Find outlet basins
        mask = maindown == 0
        if only_subid:
            outdata = subid[mask]
        else:
            outdata = self.data.loc[mask, :]
        return outdata

    def find_allupstream_subids(self, basin, only_subid=False):
        """
        Returns an array with subids of all the upstream basins from a starting
        basin. Base basin is included in the results.
        """

        # Extract all header basins
        header = self.find_header_basins(True)
        # Downstream basins from input basin
        downbasins = self.find_alldownstream_subids(basin, True)
        downbasins = downbasins[downbasins != basin]
        # Get downstream basins of header basins
        upstreams = _np.array([], dtype=int)
        for i in header:
            route = self.find_alldownstream_subids(i, True)
            if basin in route:
                upstreams = _np.hstack((
                    upstreams,
                    route
                ))
        # Get unique indexes
        upstreams = _np.unique(upstreams)
        # Remote downstreams basins from current basin
        if len(downbasins) > 0:
            upstreams = upstreams.tolist()
            for val in downbasins:
                if val in upstreams:
                    upstreams.remove(val)
            upstreams = _np.array(upstreams, dtype=int)

        if only_subid:
            outdata = upstreams
        else:
            subid1 = index_numeration(
                upstreams,
                self.data.loc[:, 'subid'].values
            )
            outdata = self.data.iloc[subid1, :]
        return outdata

    def find_alldownstream_subids(self, basin, only_subid=False):
        """
        Returns an array with subids of all the downstream basins from a starting
        basin. Base basin is included in the results.
        """

        # Convert input types
        basin = int(basin)
        subid = self.data['subid'].values
        maindown = self.data['maindown'].values
        n = subid.size  # number ob basins
        # Renumerate subids
        subid1, maindown1 = subid_numeration(subid, maindown)
        # Initial basin code
        basin0 = _np.where(subid == basin)[0][0]  # get new basin index
        basins = [basin0]
        for i in range(n):
            basin1 = maindown1[basins[-1]]
            if basin1 == -1:
                break
            else:
                basins.append(basin1)
        basins = _np.array(basins, dtype=int)
        if only_subid:
            outdata = subid[basins]
        else:
            outdata = self.data.iloc[basins, :]
        return outdata

    def upstream_area(self, inplace=False):
        """
        Computes the total area of all upstream basins from the current basin.
        Current basin is also considered in the total area.

        INPUTS:
            inplace   [bool] if False (default), GeoData is overwritten with a column
                       named 'uparea'. If True, a DataFrame with ['subid', 'maindown',
                       'area', 'uparea'] is returned
        """
        # Get subid
        subid = self.data['subid'].values
        n = subid.size  # number ob basins
        # Compute upstream area for each basins
        areas = _np.zeros(n)
        for i in range(n):
            basins_data = self.find_allupstream_subids(subid[i])
            areas[i] = basins_data['area'].sum()
        if inplace:
            self.data['uparea'] = areas
        else:
            outdata = self.data.loc[:, ['subid', 'maindown', 'area']]
            outdata['uparea'] = areas
            return outdata

    def downstream_area(self, inplace=False):
        """
        Computes the total area of all downstream basins from the current basin.
        Current basin is also considered in the total area.

        INPUTS:
            inplace   [bool] if False (default), GeoData is overwritten with a column
                       named 'downarea'. If True, a DataFrame with ['subid', 'maindown',
                       'area', 'downarea'] is returned
        """
        # Get subid
        subid = self.data['subid'].values
        n = subid.size  # number ob basins
        # Compute downstream area for each basins
        areas = _np.zeros(n)
        for i in range(n):
            basins_data = self.find_alldownstream_subids(subid[i])
            areas[i] = basins_data['area'].sum()
        if inplace:
            self.data['downarea'] = areas
        else:
            outdata = self.data.loc[:, ['subid', 'maindown', 'area']]
            outdata['downarea'] = areas
            return outdata

    def count_upstream_basins(self, inplace=False):
        """
        Counts the number of upstream basins from the current basins.
        Current basin is not considered in the count.

        INPUTS:
            inplace   [bool] if False (default), GeoData is overwritten with a column
                       named 'upbasins'. If True, a DataFrame with ['subid', 'maindown',
                       'upbasins'] is returned

        """
        # Get basin subid
        subid = self.data['subid'].values
        n = subid.size  # number ob basins
        # Count upstream basins
        count = _np.zeros(n, dtype=int)
        for i in range(n):
            basins_data = self.find_allupstream_subids(subid[i], True)
            count[i] = len(basins_data) - 1
        if inplace:
            self.data['upbasins'] = count
        else:
            outdata = self.data.loc[:, ['subid', 'maindown']]
            outdata['upbasins'] = count
            return outdata

    def count_downstream_basins(self, inplace=False):
        """
        Counts the number of downstream basins from the current basins.
        Current basin is not considered in the count.

        INPUTS:
            inplace   [bool] if False (default), GeoData is overwritten with a column
                       named 'downbasins'. If True, a DataFrame with ['subid', 'maindown',
                       'downbasins'] is returned

        """
        # Get subid
        subid = self.data['subid'].values
        n = subid.size  # number ob basins
        # Count downstream basins
        count = _np.zeros(n, dtype=int)
        for i in range(n):
            basins_data = self.find_alldownstream_subids(subid[i], True)
            count[i] = len(basins_data) - 1
        if inplace:
            self.data['downbasins'] = count
        else:
            outdata = self.data.loc[:, ['subid', 'maindown']]
            outdata['downbasins'] = count
            return outdata

    def sort_basins(self, inplace=False):
        """
        Sort basins following the requirements of HYPE model, then header basins are
        sorted first, next basins are sorted with respect the downstream order.
        Outlet basins are sorted at the end of the numeration.
        """
        # Compute number of all upstream basins
        upbasins = self.count_upstream_basins(False)
        upbasins = upbasins.sort_values('upbasins', ascending=True)
        # Find outlet basins
        mask = upbasins['maindown'] == 0
        # Sort basins
        basins_sort = _pd.concat((upbasins[~mask], upbasins[mask]))
        # Return data
        if inplace:
            self.data = (self.data.loc[basins_sort.index, :].
                         reset_index(drop=True))
        else:
            return self.data.loc[basins_sort.index, :].reset_index(drop=True)

    def outlet_basins_regions(self, inplace=False):
        """
        Identify parameter regions considering all the upstream basins from
        each outlet basin, then, number of regions is equal to number of outlet
        basins.


        INPUTS:
            inplace   [bool] if False (default), GeoData is overwritten. If True,
                       a DataFrame with all parameters is returned
        """

        # Find outlet basins
        outlets = self.find_outlet_basins(True)
        # Find all upstream basins for each outlet basin
        nr = len(outlets)  # number of regions
        regions = _np.arange(1, nr+1, dtype=int)
        data = self.to_frame().set_index('subid')
        for i in range(nr):
            upbasins = self.find_allupstream_subids(outlets[i], True)
            data.loc[upbasins, 'region'] = regions[i]
            data.loc[upbasins, 'parreg'] = regions[i]
        data.reset_index(inplace=True)
        if inplace:
            self.data = data
        else:
            return data

    def to_frame(self, columns=None):
        """
        Returns the GeoData as a DataFrame
        """
        if columns is None:
            return self.data.copy()
        else:
            return self.data.loc[:, columns]

    def save_table(self, filename):
        """
        Exports the GeoData fields to a delimited file.
        Index is ignored
        """
        if filename.endswith('.csv'):
            self.data.to_csv(filename, sep=',', index=False)
        else:
            self.data.to_csv(filename, sep='\t', index=False)

    def remove_geodataclass(self):
        """Removes all the soil-land percentajes from the GeoData"""
        columns = [x for x in self.data.columns if str(x).startswith('slc_')]
        if columns:
            self.data.drop(labels=columns, axis=1, inplace=True)

    def add_geodataclass(self, geodataclass):
        """Add or update the soil-land clasess"""
        # Read geodataclass
        if type(geodataclass) is str:
            if geodataclass.lower().endswith('.csv'):
                geodataclass = _pd.read_csv(geodataclass, delimiter=',', index_col=[0])
            else:
                geodataclass = _pd.read_csv(geodataclass, delimiter='\t', index_col=[0])
        elif type(geodataclass) is not CLASS1:
            raise TypeError('Wrong geodataclass input type.')

        # Remove current clasess
        self.remove_geodataclass()
        # Add new geoclass
        if 'subid' in geodataclass:
            self.data = self.data.merge(geodataclass, how='left', on='subid')
        else:
            self.data = self.data.merge(geodataclass, how='left',
                                        left_on='subid', right_index=True)

    def check_geodataclass(self):
        """Check that soil-land percents sums 1 to avoid problems with HYPE"""
        columns = [x for x in self.data.columns if str(x).startswith('slc_')]
        if columns:
            geodataclass = self.data.loc[:, columns]
            for i in range(geodataclass.shape[0]):
                mask = geodataclass.iloc[i, :] > 0

                # fist fix
                diff = 1 - geodataclass.iloc[i, :].sum()
                geodataclass.iloc[i][mask] += _np.round(diff / _np.sum(mask), 4)

                # second fix
                diff = 1 - geodataclass.iloc[i, :].sum()
                geodataclass.iloc[i, _np.where(mask.values)[0][0]] += diff
            geodataclass = geodataclass.round(4)
            self.data.loc[:, columns] = geodataclass

    def get_geodata(self):
        """Returns the geodata without geodataclass as a DataFrame"""
        columns = [x for x in self.data.columns if not str(x).startswith('slc_')]
        return self.data.loc[:, columns]

    def get_geodataclass(self):
        """Returns the geodataclass as a DataFrame"""
        columns = [x for x in self.data.columns if str(x).startswith('slc_')]
        if columns:
            columns = ['subid'] + columns
            return self.data.loc[:, columns]
        else:
            return _pd.DataFrame([], index=self.data['subid'].values)


# ==============================================================================
# Flow connection functions
# ==============================================================================

@_nb.jit
def index_numeration(basins, subid):
    """
    Renumerate an array of basins using a subid array
    """
    n = len(basins)
    newsubid = _np.arange(len(subid), dtype=int)
    basinsid = _np.zeros(n, dtype=int)
    for i in range(n):
        basinsid[i] = newsubid[subid == basins[i]][0]
    return basinsid


@_nb.jit
def subid_numeration(subid, maindown):
    """
    Re-numerates basins from 0 to number of basins.
    It is useful when basins contains high subids

    INPUTS:
        subid       [array] input subid
        maindown    [array] input maindown

    OUTPUTS:
        subid       [array] new subid codes
        maindown    [array] new maindown codes
    """

    # Re-numeration of subid and maindown
    n = len(subid)
    subid1 = _np.arange(n, dtype=int)
    maindown1 = _np.full(n, -1, dtype=int)
    for i in range(n):
        for j in range(n):
            if subid[i] == maindown[j]:
                maindown1[j] = i  # get downstream index
                continue
    return subid1, maindown1


@_nb.jit
def downstream_id(subid, maindown):
    """
    Connectivity matrix for downstream basins using subid and
    maindown
    """

    n = len(subid)
    # Re-numeration of subid and maindown
    subid1, maindown1 = subid_numeration(subid, maindown)
    # Create downstream matrix
    downstreams = _np.zeros((n, n), dtype=int)
    for i in range(n):
        if maindown1[i] != -1:
            downstreams[i, maindown1[i]] = 1
    return downstreams


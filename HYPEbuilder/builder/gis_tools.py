"""
==============================================================================
HYPE hydrological model tools for python

GIS tools for basins delimitation
    Layer processing tools
    DEM processing tools
    Basins delimitation
    Basins outlets
    Soil and Land Cover tools


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

from .geodata import GeoData as _GeoData
import tempfile as _tmp

from pysaga.data_manager.grids import GridObj as _GridObj
import pysaga.tools.grids as _gs
import pysaga.tools.projection as _proj
import pysaga.tools.terrain_analysis as _ta
import pysaga.tools.shapes as _sp
import pysaga.tools.geostatistics as _gstats

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

CLASS1 = type(_pd.DataFrame())
CLASS2 = type(_gpd.GeoDataFrame())


# ==============================================================================
# Import layers
# ==============================================================================

def grid_transformation(outgrid, grid, proj4, polygon=None, cellsize=0,
                        transform=True, resampling=0):
    """
    Reproject grids to the selected coordinate reference system (proj4).
    A polygon can be used to clip reprojected raster.


    INPUTS:
        outgrid       [string] output grid file name (.sgrd) with project crs
        grid          [string] input grid file name (.sgrd or .tif)
        proj4         [string] proj4 coordinate reference system format
        polygon       [string] optional input polygon used to clip grid.
                        crs of polygon must be the same than input grid
        cellsize      [int] output grid resolution in output projected coordinate system.
                        If cellsize is zero, cellsize is automatic calculated using the
                        output projection
        transform     [bool] if True (default) input grid is reprojected (recomended)
        resampling    [int] interpolation method for grid resampling
                        [0] Nearest Neigbhor (default)
                        [1] Bilinear Interpolation
                        [2] Inverse Distance Interpolation
                        [3] Bicubic Spline Interpolation
    """

    assert 0 <= resampling <= 3, 'resampling method must be between 0 and 3'

    # Try to clip raster with polygon
    if type(polygon) is str:
        print('Clipping grid with polygon...')
        _sp.clip_grid_with_polygons(
            outgrid,
            grid,
            polygon,
            extent=2)
        grid = outgrid

    # Transform or resampling grid
    if transform:  # grid transform
        print(f'Grid projection using proj4: {proj4}')
        _proj.grid_transformation(
            outgrid,
            grid,
            resampling=resampling,
            proj=proj4,
            cellsize=cellsize,
        )

    else:  # resampling
        print('Resampling grid...')
        _gs.resampling(
            outgrid,
            grid,
            scale_up=resampling,
            scale_down=resampling,
            cellsize=cellsize,
        )
    print('Finish!')


def share_grid_extent(grids, base_grid, scale_up=5, scale_down=3):
    """
    Set the same grid extent and resolution for a list of grids using
    a grid as base.
    This tool overwrites the input grids files.
    Input grids and base grid must have the same crs.


    INPUTS:
        grids         [string, list, tuple] input grids to resample.
                       grids are overwritten.
        basegrid      [string] input grid to use as grid extent
        scale_up      [int] upscaling grid method
                        [0] Nearest Neighbour
                        [1] Bilinear Interpolation
                        [2] Bicubic Spline Interpolation
                        [3] B-Spline Interpolation
                        [4] Mean Value
                        [5] Mean Value (cell area weighted)
                        [6] Minimum Value
                        [7] Maximum Value
                        [8] Majority
        scale_down    [int] downscaling grid method
                        [0] Nearest Neighbour
                        [1] Bilinear Interpolation
                        [2] Bicubic Spline Interpolation
                        [3] B-Spline Interpolation
    """

    assert 0 <= scale_up <= 8, 'wrong up scale method'
    assert 0 <= scale_down <= 3, 'wrong down scale method'

    # Check inputs
    if type(grids) is str:
        grids = [grids]
    elif type(grids) not in (list, tuple):
        raise TypeError('grids must be a grid file name or a list/tuple of grids!')
    # Resampling grids to the same grid extent
    for grid in grids:
        _gs.resampling(
            grid,
            grid,
            scale_up=scale_up,
            scale_down=scale_down,
            grid_extent=base_grid
        )


def grids_mask(grids, mask, aspolygon=False):
    """
    Set a mask to input grids. Mask can be defined as a raster or
    as a polygon file.
    This tool overwrites the input grids files.
    Input grids and mask must have the same crs.

    INPUTS:
        grids         [string, list, tuple] input grids to mask.
                       grids are overwritten.
        mask          [string] grid or vector file to use as mask
        aspolygon     [bool] if False (default), input mask is a grid.
                       In other case, mask is a vector file (.shp)
    """

    # Check inputs
    if type(grids) is str:
        grids = [grids]
    elif type(grids) not in (list, tuple):
        raise TypeError('grids must be a grid file name or a list/tuple of grids!')
    # Convert polygon to raster
    if aspolygon:
        newfile = _os.path.join(_tmp.gettempdir(), 'mask.sgrd')
        _gs.shapes_to_grid(
            newfile,
            mask,
            grid_extent=grids[0]
        )
        mask = newfile
    # Set mask
    for grid in grids:
        _gs.masking(
            grid,
            grid,
            mask)


# ==============================================================================
# Terrain process
# ==============================================================================

def dem_processing(folder, basename, dem, streams=None, minslope=0.01, epsilon=5):
    """
    Terrain processing using a DEM as input
        Burn streams into DEM (optional)
        Fill DEM. removes sinks on DEM
        Flow accumulation. detects pixels with higher flow
        Flow direction (D8). topographic flow direction using single algorithm
        Slope and Aspect. terrain properties needed


    INPUTS:
        folder        [string] folder where outputs will be saved
        basename      [string] base name for all output grids
        dem           [string] input dem grid
        streams       [string] grid or shape file of streams to burn into dem
        minslope      [float] minimum slope (degress) to preserve from cell to cell
        epsilon       [float] parameter to decrease cell values in streams burn algorithm

    OUTPUTS:
        outputs       [dict] output files dictionary
                        'dem': filled dem
                        'flowdir': flow direction
                        'flowacc': flow accumulation
                        'slope': dem slope in percent

    """

    # Create ouput file names
    print('Getting file names...')
    outputs = {
        'dem': _os.path.join(folder, f'{basename}_DemFilled.sgrd'),
        'flowdir': _os.path.join(folder, f'{basename}_FlowDir.sgrd'),
        'flowacc': _os.path.join(folder, f'{basename}_FlowAcc.sgrd'),
        'slope': _os.path.join(folder, f'{basename}_Slope.sgrd')
    }
    # temporary files
    watersheds = _os.path.join(_tmp.gettempdir(), 'watersheds.sgrd')
    burned_dem = _os.path.join(_tmp.gettempdir(), 'dem_burned.sgrd')
    aspect = _os.path.join(_tmp.gettempdir(), 'aspect_rad.sgrd')

    # Check is streams will be burn in dem
    if type(streams) is str:
        print('Burning streams into dem...')
        _ta.burn_stream_network_into_dem(
            burned_dem,
            dem,
            streams,
            epsilon=epsilon
        )
        dem = burned_dem

    # Fill dem and flow directions computing
    print('Filling sinks and computing flow directions...')
    _ta.fill_sinks_wangliu(
        dem,
        outdem=outputs['dem'],
        outflowdir=outputs['flowdir'],
        outwshed=watersheds,
        minslope=minslope
    )

    # Flow accumulation computation
    print('Computing flow accumulation...')
    _ta.flow_accumulation(
        outputs['flowacc'],
        outputs['dem'],
        sinkroute=outputs['flowdir']
    )

    # Compute slope
    print('Computing terrain slope...')
    _ta.dem_slope_aspect(
        outputs['dem'],
        outputs['slope'],
        aspect,
        method=1,
        sunits=2,
        aunits=0
    )
    print('Finish!')
    return outputs


def channel_network(folder, basename, dem, flowdir, flowacc, init_value=1000, min_len=100):
    """
    Defines the channel network in raster and vector format for basins
    delimitation and network properties estimation.
    This tools also generates points that are used as basins outlets
    for the basins delimitation. Points can be edited with a GIS to
    add or remove basins' outlet. At least, one outlet must be conserved
    for the basins delimitation. More basin outlets must be defined over
    the channel network to avoid flow accumulation conflicts.

    INPUTS:
        folder        [string] folder where outputs will be saved
        basename      [string] base name for all output grids
        dem           [string] input filled dem grid
        flowdir       [string] input flow direction grid
        flowacc       [string] input flow accumulation grid
        init_value    [int] minimum number of cells from flow accumulation
                       raster to start the channel network.
        min_len       [int] minimum streams length in m

    OUTPUTS:
        outputs       [dict] output files dictionary
                        'streams': vector file of channels
                        'channels': grid file of channels
                        'channelsdir': grid file of channels flow direction
                        'outlets': vector file of basins outlets
    """

    if not _os.path.exists(dem):
        raise IOError(f'dem < {dem} > does not exist')
    if not _os.path.exists(flowdir):
        raise IOError(f'flowdir < {flowdir} > does not exist')
    if not _os.path.exists(flowacc):
        raise IOError(f'flowacc < {flowacc} > does not exist')

    # Output layers
    outputs = {
        'streams': _os.path.join(folder, f'{basename}_Streams.shp'),
        'channels': _os.path.join(folder, f'{basename}_Channels.sgrd'),
        'channelsdir': _os.path.join(folder, f'{basename}_ChannelsDir.sgrd'),
        'outlets': _os.path.join(folder, f'{basename}_BasinsOutlets.shp')
    }

    # Channel network
    print('Computing channel networks...')
    _ta.channel_network(
        dem,
        flowdir=flowdir,
        init_grid=flowacc,
        channels=outputs['streams'],
        gridchannels=outputs['channels'],
        gridchandir=outputs['channelsdir'],
        init_value=init_value,
        init_method=2,
        min_len=min_len
    )

    # Get basins outlets as points
    print('Extracting basins outlets...')
    layer = _GridObj(outputs['channels'])

    data = layer.get_data(1)
    outlets = _np.where(data == -1)  # find basins outlet
    pixels = list(zip(outlets[0], outlets[1]))  # find outlets in grid
    coors = layer.pixel2coor(pixels)  # convert pixels to points

    # reclassify channel values
    data[data >= -1] = 1
    layer.set_data(data)
    layer.close()

    # Convert coordinates to vector points
    df = _pd.DataFrame(
        coors,
        columns=['X', 'Y'],
        index=_np.arange(1, len(coors) + 1, dtype=int)
    )
    df.index.name = 'ID'
    df['Name'] = ['outlet' + str(x) for x in df.index]
    tempfile = _os.path.join(_tmp.gettempdir(), 'points.csv')
    df[['Name', 'X', 'Y']].to_csv(tempfile)

    _sp.convert_table_to_points(
        outputs['outlets'],
        tempfile,
        'X',
        'Y',
    )

    _proj.set_crs(
        shapes=outputs['outlets'],
        crs_method=1,
        proj=dem
    )

    print('Finish!')
    return outputs


# ==============================================================================
# Basins delimitation
# ==============================================================================

def basins_delimitation(folder, basename, outlets, dem, flowdir, slope,
                        channels, streams, proj4, find_regions=False):
    """
    Basins delineation and parameters estimation for the HYPE model
    Basins are computed from the outlet basins points, for instance,
    user can add or delete new basins modifying the outlets layer


    INPUTS:
        folder        [string] folder where outputs will be saved
        basename      [string] base name for all output grids
        outlets       [string] input outlet basins vector file
        dem           [string] input dem grid file
        flowdir       [string] input flow direction grid file
        slope         [string] input slope in percent grid file
        channels      [string] input channels grid file
        streams       [string] input channels vector file
        proj4         [string] input layers projection
        find_regions  [bool] if True, basins with same output basins
                       is set in the same region

    OUTPUTS:
        outputs       [dict] output files dictionary
                        'gridbasins': subbasins grid file
                        'basins': subbasins vector file
                        'geodata': GeoData table in csv format

    """
    # Get files
    if not _os.path.exists(outlets):
        raise IOError(f'basins outlet < {outlets} > does not exist')
    if not _os.path.exists(dem):
        raise IOError(f'dem < {dem} > does not exist')
    if not _os.path.exists(flowdir):
        raise IOError(f'flowdir < {flowdir} > does not exist')
    if not _os.path.exists(slope):
        raise IOError(f'slope < {slope} > does not exist')
    if not _os.path.exists(channels):
        raise IOError(f'grid channels < {channels} > does not exist')
    if not _os.path.exists(streams):
        raise IOError(f'vector channels < {streams} > does not exist')

    # Output files
    outputs = {
        'gridbasins': _os.path.join(folder, f'{basename}_GridBasins.sgrd'),
        'basins': _os.path.join(folder, f'{basename}_Basins.shp'),
        'geodata': _os.path.join(folder, f'{basename}_GeoData.csv'),
    }

    # Set outlet grids
    print('Getting basins outlets as pixels...')
    channels_mod = _os.path.join(_tmp.gettempdir(), 'channels.sgrd')
    _gs.copy_grid(channels_mod, channels)

    layer = _GridObj(channels_mod)

    # extract points coordinates
    points = _gpd.read_file(outlets)
    coors = list(zip(points.geometry.x, points.geometry.y))
    pixels = layer.coor2pixel(coors)
    pixels = _np.array(pixels, dtype=int)
    if pixels.ndim == 1:
        pixels = _np.array([pixels], dtype=int)

    # set outlets over channels
    data = layer.get_data()
    data[pixels[:, 0], pixels[:, 1]] = -1
    layer.set_data(data)

    layer.close()
    del(data)

    # Watershed basins delimitation
    print('Basins delimitation using outlets...')
    _ta.watershed_basins(
        outputs['gridbasins'],
        dem,
        channels_mod,
        minsize=0,
        sinkroute=flowdir
    )

    # Convert watersheds grids to polygons
    _sp.vectorising_grid_classes(
        outputs['basins'],
        outputs['gridbasins'],
        method=1
    )

    # Clean polygons
    clean_basins_polygons(outputs['basins'], outputs['basins'])

    # Get downslope basin
    print('Getting downstream basisns codes...')
    flowdir_layer = _GridObj(flowdir)
    flowdir_matrix = flowdir_layer.get_data()
    flowdir_layer.close()

    subbasins_layer = _GridObj(outputs['gridbasins'])
    subbasins_matrix = subbasins_layer.get_data()
    subbasins_layer.close()

    # current basin
    currbasin = subbasins_matrix[pixels[:, 0], pixels[:, 1]]
    position = ~_np.isnan(currbasin)  # found no data pixels

    # ignore nans
    currbasin = currbasin[position]
    pixels = pixels[position, :]

    if len(currbasin) == 0:
        raise ValueError('Basins where not found! check your inputs.')

    # downstream basins
    downbasin = flowdir_values(
        pixels,
        flowdir_matrix,
        subbasins_matrix,
        outlets=0
    )
    del(flowdir_matrix)
    del(subbasins_matrix)

    # Compute elevation and slope statistics
    print('Computing elevation and slope statistics...')
    basins_properties = _os.path.join(_tmp.gettempdir(), 'basins.shp')
    _sp.grid_statistics_for_polygons(
        basins_properties,
        outputs['basins'],
        [dem, slope],
        method=0,
        naming=0,
        stats=['mean', 'std']
    )

    # Read GeoDataFrames
    layer1 = _gpd.read_file(outputs['basins'])
    layer2 = _gpd.read_file(basins_properties)

    # Set downstream basins
    downstreams = _pd.DataFrame(
        list(zip(currbasin, downbasin)),
        columns=['VALUE', 'maindown']
    )
    layer1 = layer1.merge(downstreams, on='VALUE')

    # Add name column
    layer1['name'] = layer1['VALUE'].apply(
        lambda x: f'basin_{x:.0f}'
    )

    # Join basins with properties
    layer1 = layer1[[
        'geometry',
        'ID',
        'VALUE',
        'maindown',
        'name'
    ]].merge(layer2[[
        'ID',
        'G01_MEAN',
        'G01_STDDEV',
        'G02_MEAN',
        'G02_STDDEV',
    ]],
             on='ID')

    # Rename columns
    layer1.columns = [
        'geometry',
        'id',
        'subid',
        'maindown',
        'name',
        'elev_mean',
        'elev_std ',
        'slope_mean',
        'slope_std'
    ]
    layer1.drop('id', axis=1, inplace=True)

    # Compute area and perimeter
    print('Computing basins properties...')
    n = layer1.shape[0]
    areas = _np.zeros(n)
    perimeters = _np.zeros(n)
    for i in range(n):
        areas[i] = layer1.iloc[i].geometry.area
        perimeters[i] = layer1.iloc[i].geometry.length
    layer1['area'] = areas
    layer1['perimeter'] = perimeters
    # Compute river length
    temp_file = _os.path.join(_tmp.gettempdir(), 'drainage.shp')
    _sp.line_dissolve(
        temp_file,
        streams,
        fields=None  # dissolve all lines
    )

    layer2 = _gpd.read_file(temp_file)
    riv_lens = _np.zeros(n)
    for i in range(layer1.shape[0]):
        try:
            riv_lens[i] = (layer1.iloc[i].
                           geometry.
                           intersection(layer2.iloc[0].geometry).
                           length)
        except:
            print("A Topology error was found when computing river length")
    layer1['rivlen'] = riv_lens

    # Compute basins centroids
    centroids = layer1.geometry.geometry.centroid
    coors = []
    for val in centroids:
        coors.append([val.x, val.y])  # extract centroid coordenates
    # project points to wgs84

    coors = _proj.reproject_points(coors,
                                   proj4,
                                   '+proj=longlat +datum=WGS84 +no_defs',
                                   stype=2)
    coors = _np.array(coors)
    layer1['latitude'] = _np.round(coors[:, 1], 4)

    # set general parameters
    layer1['region'] = 1
    layer1['parreg'] = 1
    layer1['petmodel'] = 0

    # Fix data types
    key_int = ['subid', 'maindown', 'region', 'parreg', 'petmodel']
    for key in key_int:
        layer1[key] = layer1[key].astype(int)

    key_float = ['elev_mean', 'elev_std ', 'slope_mean', 'slope_std',
                 'area', 'perimeter', 'rivlen']
    for key in key_float:
        layer1[key] = _np.round(layer1[key], 2)

    # Sorting basins following flow direction
    gd = _GeoData(layer1)
    gd.sort_basins(True)
    if find_regions:
        print('Searching parameter regions for outlet basins...')
        gd.outlet_basins_regions(True)
        data = gd.to_frame(['subid', 'region', 'parreg'])
        layer1.set_index('subid', inplace=True)
        layer1.loc[data.loc[:, 'subid'].values, 'region'] = data.loc[:, 'region'].values
        layer1.loc[data.loc[:, 'subid'].values, 'parreg'] = data.loc[:, 'parreg'].values
        layer1.reset_index(inplace=True)

    print('Exporting GeoData table...')
    gd.save_table(outputs['geodata'])
    # Save subbasins layer
    print('Saving subbasins vector layer...')
    layer1.to_file(outputs['basins'])
    print('Finish!')
    return outputs


def merge_basins(saveas, basins):
    """
    Merge no overlaping basins in a new shapefile

    Args:
        saveas:     [string] output shape file
        basins:     [list, tuple, array] input list of basins shape files

    """

    geoframes = []
    for basin in basins:
        if type(basin) is str:
            geoframes.append(_gpd.read_file(basin))
        elif type(basin) is CLASS2:
            geoframes.append(basin)
        else:
            raise TypeError('Wrong input argument in basins list')

    cnt = 1
    for i, layer in enumerate(geoframes):
        # change id numbers
        outlets = layer['maindown'].values == 0
        subid, maindown = index_renumeration(
            layer['subid'].values,
            layer['maindown'].values
        )
        layer['subid'] = subid + cnt
        layer['maindown'] = maindown + cnt
        layer.loc[outlets, 'maindown'] = 0

        if i == 0:
            merged = layer.copy()
        else:
            merged = merged.append(layer)

        cnt += len(layer)

    merged.to_file(saveas)
    print(f'Basins merged on {saveas}')


def flowdir_values(pixels, flowdir, values, outlets=0):
    """
    Returns values of input pixels over a matrix using the
    flow direction matrix.


    INPUTS:
        pixels      [array] input pixels [[x1,y2],[x2,y2]]
        flowdir     [array] flow direction matrix
        values      [array] matrix of values to extract
        outlets     [int, float] value to use for outlet pixels

    OUTPUTS:
        array       [array] values following the flow direction
    """

    # Get number of rows and columns
    r, c = flowdir.shape
    r1, c1 = r - 1, c - 1

    # define flow direction x,y increases
    routine = {
        0: (-1, 0),   # N
        1: (-1, 1),   # NE
        2: (0, 1),    # E
        3: (1, 1),    # SE
        4: (1, 0),    # S
        5: (1, -1),   # SW
        6: (0, -1),   # W
        7: (-1, -1),  # NW
        8: (-1, 0),   # N
        -1: (0, 0)    # outlet
    }

    # pixels increases
    n = pixels.shape[0]
    outvalues = _np.zeros(n)
    for i in range(n):
        dx, dy = routine.get(
            flowdir[pixels[i, 0], pixels[i, 1]],
            (0, 0)
        )
        x, y = pixels[i, :]

        if dx == 0 and dy == 0:  # flowdir == -1
            outvalues[i] = outlets
        elif ((x + dx) < 0 or (x + dx) > r1 or
              (y + dy) < 0) or (y + dy) > c1:  # out of range
            outvalues[i] = outlets
        else:
            outvalues[i] = values[x + dx, y + dy]

        if _np.isnan(outvalues[i]):
            outvalues[i] = outlets
    return outvalues


# ==============================================================================
# Soil and Land Cover tools
# ==============================================================================

def usda_soil_texture(outable, outgrid, sand, silt, clay):
    """
    Soil texture classification using USDA method.

    INPUTS:
        outable    [string] output table file name (.csv) for soil types derived
                    from USDA classification
        outgrid    [string] output soil texture grid (.sgrd)
        sand       [string] input grid of sand content in percent
        silt       [string] input grid of silt content in percent
        clay       [string] input grid of clay content in percent
    """

    # Soil texture classification
    _gs.soil_texture_classification(
        outgrid,
        sand,
        silt,
        clay,
        method=0
    )
    # Create output classification tables
    classes = [
        [1, 'C', 'Clay'],
        [2, 'SiC', 'Silt Clay'],
        [3, 'SiCL', 'Silty Clay Loam'],
        [4, 'SC', 'Sandy Clay'],
        [5, 'SCL', 'Sandy Clay Loam'],
        [6, 'CL', 'Clay Loam'],
        [7, 'Si', 'Silt'],
        [8, 'SiL', 'Silt Loam'],
        [9, 'L', 'Loam'],
        [10, 'S', 'Sand'],
        [11, 'LS', 'Loamy Sand'],
        [12, 'SL', 'Sandy Loam']
    ]
    soil = _pd.DataFrame(
        classes,
        columns=['soil_type', 'soil_name', 'soil_desc']
    )
    soil.to_csv(outable, index=False)


def soil_reclass(folder, basename, soil_grid, soil_rules=None, other=None,
                 threshold=0, new_val=1):
    """
    Grid reclassification using rules or threshold values
    to remove low frequency soil classes.

    INPUTS:
        folder         [string] folder where outputs will be saved
        basename       [string] base name for all output grids
        soil_grid      [string] input soil texture grid file
        soil_rules     [string, DataFrame] optional rules to reclassify input grid.
                        Rules can be a csv file with three columns ['new', 'min', 'max'],
                        where new is the new value, min and max is the range of the old
                        value. Rules can be a DataFrame with the same columns than the csv
                        file.
        other          [int] if soil_rules parameter is used, other is used to classify all
                        the other values in the grid excluded from the rules.
        threshold      [int] optional threshold percent to change pixel categories of low frequency.
                        i.e. soil categories that covers an area lower than threshold are changed.
        new_val        [int] if threshold is different that zero, new_val is used to reclassify the
                        low frequency categories
    """

    # Output files
    outputs = {
        'soil': _os.path.join(folder, f'{basename}_soil.sgrd'),
        'soil_rules': _os.path.join(folder, f'{basename}_soil_rules.csv'),
    }

    # Check if soil_rules is input
    if type(soil_rules) is str:
        soil_rules = _pd.read_csv(soil_rules)
    if type(soil_rules) is CLASS1:
        soil_rules.columns = [str(x).lower() for x in soil_rules.columns]
        soil_rules = soil_rules[['new', 'min', 'max']].values

        # reclassify grid values
        print('Soil grid reclassification...')
        _gs.reclassify_values(
            outputs['soil'],
            soil_grid,
            single=None,
            vrange=soil_rules,
            smethod=0,
            rmethod=1,
            other=other,
            nodata=-9999
        )

    elif soil_rules is None:
        _gs.copy_grid(
            outputs['soil'],
            soil_grid
        )
    else:
        raise TypeError('Bad argument type for soil_rules!')

    # Load layer and compute
    layer = _GridObj(outputs['soil'])
    data = _np.array(layer.get_data(), dtype=int)
    # Find unique values
    vals, count = _np.unique(data, return_counts=True)
    # remove nans
    posna = _np.isnan(vals)
    vals = vals[~posna]
    count = count[~posna]

    # Reduce classes with low frequency
    if 0 < threshold < 100:
        print('Removing low frequency classes...')
        # compute percent
        percent = count / _np.sum(count) * 100.
        cnt = 0
        for i in range(len(percent)):
            if percent[i] < threshold:
                cnt += 1
                data[data == vals[i]] = new_val
        if cnt > 0:
            layer.set_data(data)

        # Recompute unique values
        vals, count = _np.unique(data, return_counts=True)
        # remove nans
        posna = _np.isnan(vals)
        vals = vals[~posna]
    layer.close()
    # Output table of unique values
    table = _pd.DataFrame(vals, columns=['id'])
    table['id'] = table['id'].astype(int)
    table['code'] = _np.nan
    table['name'] = ['c' + str(int(x)) for x in vals]
    table['description'] = ['class' + str(int(x)) for x in vals]
    table.to_csv(outputs['soil_rules'], index=False)

    print('Finish!')
    return outputs


def land_reclass(folder, basename, land_grid, land_rules=None, other=None,
                 threshold=0, new_val=1):
    """
    Grid reclassification using rules or threshold values
    to remove low frequency land classes.

    INPUTS:
        folder         [string] folder where outputs will be saved
        basename       [string] base name for all output grids
        land_grid      [string] input land cover grid file
        land_rules     [string, DataFrame] optional rules to reclassify input grid.
                        Rules can be a csv file with three columns ['new', 'min', 'max'],
                        where new is the new value, min and max is the range of the old
                        value. Rules can be a DataFrame with the same columns than the csv
                        file.
        other          [int] if soil_rules parameter is used, other is used to classify all
                        the other values in the grid excluded from the rules.
        threshold      [int] optional threshold percent to change pixel categories of low frequency.
                        i.e. soil categories that covers an area lower than threshold are changed.
        new_val        [int] it threshold is different that zero, new_val is used to reclassify the
                        low frequency categories
    """

    # Output files
    outputs = {
        'land': _os.path.join(folder, f'{basename}_land.sgrd'),
        'land_rules': _os.path.join(folder, f'{basename}_land_rules.csv'),
    }

    # Check if land_rules is input
    if type(land_rules) is str:
        land_rules = _pd.read_csv(land_rules)
    if type(land_rules) is CLASS1:
        land_rules.columns = [str(x).lower() for x in land_rules.columns]
        land_rules = land_rules[['new', 'min', 'max']].values

        # reclassify grid values
        print('Land cover grid reclassification...')
        _gs.reclassify_values(
            outputs['land'],
            land_grid,
            single=None,
            vrange=land_rules,
            smethod=0,
            rmethod=0,
            other=other,
            nodata=-9999
        )

    elif land_rules is None:
        _gs.copy_grid(
            outputs['land'],
            land_grid
        )
    else:
        raise TypeError('Bad argument type for land_rules!')

    # Load layer and compute
    layer = _GridObj(outputs['land'])
    data = _np.array(layer.get_data())
    # Find unique values
    vals, count = _np.unique(data, return_counts=True)
    # remove nans
    posna = _np.isnan(vals)
    vals = vals[~posna]
    count = count[~posna]

    # Reduce classes with low frequency
    if 0 < threshold < 100:
        print('Removing low frequency classes...')
        # compute percent
        percent = count / _np.sum(count) * 100.
        cnt = 0
        for i in range(len(percent)):
            if percent[i] < threshold:
                cnt += 1
                data[data == vals[i]] = new_val
        if cnt > 0:
            layer.set_data(data)

        # Recompute unique values
        vals, count = _np.unique(data, return_counts=True)
        # remove nans
        posna = _np.isnan(vals)
        vals = vals[~posna]
    layer.close()
    # Output table of unique values
    table = _pd.DataFrame(vals, columns=['id'])
    table['id'] = table['id'].astype(int)
    table['code'] = _np.nan
    table['name'] = ['c' + str(int(x)) for x in vals]
    table['description'] = ['class' + str(int(x)) for x in vals]
    table['vegetation_type'] = 0
    table['special_class'] = 0
    table.to_csv(outputs['land_rules'], index=False)

    print('Finish!')
    return outputs


def soil_land_combinations(folder, basename, basins, soil, land, threshold=0):
    """
    Computes the Soil-Land classes combinations using the soil_texture
    and land_cover grids. This tools creates the GeoClass
    table used to define soil attributes for each combination.
    Additionally, the proportion of soil-land-class (slc) for each subbasin
    is computed and returned as a GeoDataClass file.

    INPUTS:
        folder         [string] folder where outputs will be saved
        basename       [string] base name for all output files
        basins         [string] input basins vector file
        soil           [string] input soil class grid
        land           [string] input land class grid
        threshold:     [float] percent [0-1] if different to zero, slc lower than
                         threshold value are set as zero
    OUTPUTS:
        outputs       [dict] output files dictionary
                        'soilclass': basins soil class grid
                        'landclass': basins land class grid
                        'geoclass': GeoClass table
                        'geodataclass': Soil-Land combination table for each basin
                        'codes': reclass values for soil and land

    """

    # Create output files
    outputs = {
        'soilclass': _os.path.join(folder, f'{basename}_BasinsSoilClass.sgrd'),
        'landclass': _os.path.join(folder, f'{basename}_BasinsLandClass.sgrd'),
        'geoclass': _os.path.join(folder, f'{basename}_GeoClass.csv'),
        'geodataclass': _os.path.join(folder, f'{basename}_GeoDataClass.csv'),
        'codes': _os.path.join(folder, f'{basename}_SLNewCodes.csv'),
    }

    # Temporary files
    basins_grid = _os.path.join(_tmp.gettempdir(), 'basins.sgrd'),
    zstats = _os.path.join(_tmp.gettempdir(), 'zonal_stats.csv')

    # Rasterize basins
    print('Conversion of basins from vector to grid...')
    _gs.shapes_to_grid(
        basins_grid,
        basins,
        value_method=2,
        field='subid',
        multiple_values=1,
        poly_type=1,
        data_type=8,
        grid_extent=soil
    )

    # Mask soil and land with basins
    print('Masking soil and land grids with basins as mask...')
    _gs.masking(outputs['soilclass'], soil, basins_grid)
    _gs.masking(outputs['landclass'], land, basins_grid)

    # Reclassify soil grid using numeration of 1 to n
    print('Soil and land grids reclassification...')
    grid_list = [outputs['soilclass'], outputs['landclass']]
    codes_list = []

    for j in range(len(grid_list)):
        layer = _GridObj(grid_list[j])
        data = layer.get_data()
        # change no data values
        vals = _np.unique(data)
        vals = vals[~_np.isnan(vals)]
        # reclassify values
        n = len(vals)
        new_vals = _np.arange(1, n + 1, dtype=int)
        codes_list.append(_np.vstack((vals, new_vals)).transpose())
        reclass_data = _np.full_like(data, _np.nan)
        for i in range(n):
            reclass_data[data == vals[i]] = new_vals[i]
        layer.set_data(reclass_data)
        layer.close()

    new_codes_soil = _pd.DataFrame(codes_list[0], columns=['old', 'new']).astype(int)
    new_codes_soil['grid'] = 'soil'
    new_codes_land = _pd.DataFrame(codes_list[1], columns=['old', 'new']).astype(int)
    new_codes_land['grid'] = 'land'
    codes = _pd.concat((new_codes_soil, new_codes_land), axis=0, ignore_index=True)
    codes[['grid', 'old', 'new']].to_csv(outputs['codes'], index=False)

    # Compute zonal statistics classes combination
    print('Computing Soil-Land-Combinations...')
    _gstats.zonal_grid_statistics(
        zstats,
        outputs['landclass'],
        categories=outputs['soilclass'],
        grids=None,
        aspect=None,
        shortnames=True
    )

    # Create GeoClass
    geoclass = _pd.read_csv(zstats)
    geoclass.dropna(inplace=True)  # delete null values
    geoclass = geoclass.iloc[:, :2]  # save only the first columns
    geoclass.columns = ['land_code', 'soil_code']
    geoclass = geoclass.astype(int)
    # re-numerate soil land combinations
    geoclass.index = _np.arange(1, geoclass.shape[0] + 1, dtype=int)
    geoclass.index.name = 'slc'
    # create geoclass properties
    geoclass['main_crop'] = 0
    geoclass['second_crop'] = 0
    geoclass['crop_rotation'] = 0
    # set vegetation class
    geoclass['vegetation_type'] = 1
    # set special class
    geoclass['special_class'] = 0
    # set layers and depths
    geoclass['tile_depth'] = 0
    geoclass['stream_depth'] = 1
    geoclass['no_layers'] = 2
    geoclass['depth_1'] = 0.5
    geoclass['depth_2'] = 1.0
    geoclass['depth_3'] = 0.0
    # save geoclass
    geoclass.to_csv(outputs['geoclass'], index=True)

    # Compute basins-soil-class combinations
    _gstats.zonal_grid_statistics(
        zstats,
        basins_grid,
        categories=[outputs['landclass'], outputs['soilclass']],
        grids=None,
        aspect=None,
        shortnames=True
    )

    # Generate GeoData-SoilLandClass
    print('Computing Soil-Land-Combinations by basin...')
    combinations = _pd.read_csv(zstats)
    combinations.dropna(inplace=True)  # delete null values
    combinations = combinations.astype(int)
    combinations.columns = ['subid', 'land', 'soil', 'count']
    combinations.set_index('subid', inplace=True)
    # create DataFrame of basins id and land combinations
    # basins id
    basins_id = _np.unique(combinations.index)
    # land-soil combinations according slc id
    slc_id = list(zip(geoclass['land_code'].values,
                      geoclass['soil_code'].values))
    # create column names
    colnames = [f'slc_{x}' for x in range(1, geoclass.shape[0] + 1)]
    slc_by_basin = _pd.DataFrame(
        _np.zeros((len(basins_id), len(colnames)), dtype=float),
        columns=colnames,
        index=basins_id
    )
    slc_by_basin.index.name = 'subid'
    for i in basins_id:
        # soil-land-combinations by basin
        slc = combinations.loc[i, :]
        if isinstance(slc, _pd.Series):
            slc = slc.to_frame().transpose()
        # set land and soil as indexes
        slc.set_index(['land', 'soil'], inplace=True)
        # query all land-soil classes
        slc = slc.loc[slc_id, :]
        slc = slc.fillna(0)  # replace nans by zero
        # compute are cover by the slc
        slc['cover'] = slc['count'] / slc['count'].sum()
        if threshold > 0:
            mask = slc['cover'] <= threshold / 100.
            slc.loc[mask, 'count'] = 0
            slc['cover'] = slc['count'] / slc['count'].sum()
        # save slc
        slc_by_basin.loc[i, :] = _np.round(slc['cover'].values, 4)
    slc_by_basin = slc_by_basin.round(4)

    # Threshold for low frequency class on basins
    if threshold > 0:
        freq = slc_by_basin.sum()
        percent = freq / freq.sum()
        mask = (percent <= threshold).values
        slc_by_basin.loc[:, mask] = 0

    # Fix GeoData-SoilLandClass rows = 1
    for i in range(slc_by_basin.shape[0]):
        mask = slc_by_basin.iloc[i, :] > 0

        # fist fix
        diff = 1 - slc_by_basin.iloc[i, :].sum()
        slc_by_basin.iloc[i][mask] += _np.round(diff / _np.sum(mask), 4)

        # second fix
        diff = 1 - slc_by_basin.iloc[i, :].sum()
        pos = _np.where(mask.values)[0]
        slc_by_basin.iloc[i, pos[0]] += diff
    slc_by_basin = slc_by_basin.round(4)
    slc_by_basin.to_csv(outputs['geodataclass'])

    return outputs


def clean_geoclass(folder, basename, geoclass, geodataclass, threshold=0.03):
    """
        Removes low frequency soil-land clases on geoclass and geodataclass
        using a threshold

        INPUTS:
            folder         [string] folder where outputs will be saved
            basename       [string] base name for all output files
            geoclass       [string] input geoclass table
            geodataclass   [string] input geodataclass table
            threshold:     [int] percent (0-1) of subbasin area to consider a slc

        OUTPUTS:
            outputs       [dict] output files dictionary
                            'geoclass': output geoclass table
                            'geodataclass': output geoclass table
                            'codes': output table with new soil and land codes

        """

    if type(geoclass) is str:
        if geoclass.lower().endswith('.csv'):
            geoclass = _pd.read_csv(geoclass, delimiter=',', index_col=[0])
        else:
            geoclass = _pd.read_csv(geoclass, delimiter='\t', index_col=[0])
    elif type(geoclass) is not CLASS1:
        raise TypeError('Wrong geoclass input type.')

    if type(geodataclass) is str:
        if geodataclass.lower().endswith('.csv'):
            geodataclass = _pd.read_csv(geodataclass, delimiter=',', index_col=[0])
        else:
            geodataclass = _pd.read_csv(geodataclass, delimiter='\t', index_col=[0])
    elif type(geodataclass) is not CLASS1:
        raise TypeError('Wrong geodataclass input type.')

    if 'slc' in geoclass.columns:
        geoclass.set_index('slc', inplace=True)

    if 'subid' in geodataclass.columns:
        geodataclass.set_index('subid', inplace=True)

    if geodataclass.shape[1] != geoclass.shape[0]:
        raise ValueError('Number of soil-land classes must be equal in geoclass and geodataclass')

    # Create output files
    outputs = {
        'geoclass': _os.path.join(folder, f'{basename}_GeoClass.csv'),
        'geodataclass': _os.path.join(folder, f'{basename}_GeoDataClass.csv'),
        'codes': _os.path.join(folder, f'{basename}_SoilLandCodes.csv')
    }

    # Remove low frequency class
    freq = geodataclass.sum()
    percent = freq / freq.sum()
    mask = (percent > threshold).values

    # Save new clasess
    soil_code = _np.sort(_np.unique(geoclass['soil_code'].values))
    land_code = _np.sort(_np.unique(geoclass['land_code'].values))

    geoclass = geoclass.loc[mask, :]
    geodataclass = geodataclass.loc[:, mask]

    # Set new soil and land codes
    soil_old = _np.sort(_np.unique(geoclass['soil_code'].values))
    land_old = _np.sort(_np.unique(geoclass['land_code'].values))
    soil_new = _np.arange(1, len(soil_old)+1, dtype=int)
    land_new = _np.arange(1, len(land_old)+1, dtype=int)

    soil_code_new = _np.zeros_like(soil_code)
    land_code_new = _np.zeros_like(land_code)

    for i in range(len(soil_old)):
        soil_code_new[soil_code == soil_old[i]] = soil_new[i]
        geoclass.loc[geoclass['soil_code'] == soil_old[i], 'soil_code'] = soil_new[i]
    for i in range(len(land_old)):
        land_code_new[land_code == land_old[i]] = land_new[i]
        geoclass.loc[geoclass['land_code'] == land_old[i], 'land_code'] = land_new[i]

    codes = _pd.DataFrame(_np.append(soil_code, land_code), columns=['old'])
    codes['new'] = _np.append(soil_code_new, land_code_new)
    codes['kind'] = 'land'
    codes.iloc[:len(soil_code), -1] = 'soil'

    # Rename indexes
    index = _np.arange(1, geoclass.shape[0]+1, dtype=int)
    index_text = [f'slc_{x}' for x in index]
    geoclass.index = index
    geodataclass.columns = index_text
    geodataclass = geodataclass.round(4)

    # Fix geodataclass rows = 1
    for i in range(geodataclass.shape[0]):
        mask = geodataclass.iloc[i, :] > 0

        # fist fix
        diff = 1 - geodataclass.iloc[i, :].sum()
        geodataclass.iloc[i][mask] += _np.round(diff / _np.sum(mask), 4)

        # second fix
        diff = 1 - geodataclass.iloc[i, :].sum()
        geodataclass.iloc[i, _np.where(mask.values)[0][0]] += diff
    geodataclass = geodataclass.round(4)

    # Save files
    geoclass.index.name = 'slc'
    geodataclass.index.name = 'subid'

    geoclass.to_csv(outputs['geoclass'])
    geodataclass.to_csv(outputs['geodataclass'])
    codes.to_csv(outputs['codes'], index=False)

    return outputs


def clean_basins_polygons(output_layer, input_layer):
    """
    Remove mult-iparts to polygons to avoid issues with basins geoprocesing

    :param output_layer:
    :param input_layer:
    :return: None
    """
    _sp.polygon_parts(output_layer, input_layer, lakes=True)
    layer = _gpd.read_file(output_layer)
    col = layer.columns[0]
    layer["area"] = layer.area
    layer = layer.sort_values("area", ascending=False)
    layer = layer.drop_duplicates(subset=col)
    layer = layer.drop("area", axis=1)
    layer.to_file(output_layer)


# ==============================================================================
# Lakes algorithms
# ==============================================================================


# ==============================================================================
# Crop tools
# ==============================================================================


# ==============================================================================
# Branch tools
# ==============================================================================


# ==============================================================================
# Aquifer tools
# ==============================================================================


# ==============================================================================
# Utilities
# ==============================================================================

def index_renumeration(subid, maindown):
    """Set index numeration from 0 to number of basins"""
    n = len(subid)
    subid1 = _np.arange(n, dtype=int)
    maindown1 = _np.full(n, -1, dtype=int)
    for i in range(n):
        for j in range(n):
            if subid[i] == maindown[j]:
                maindown1[j] = i  # get downstream index
                continue
    return subid1, maindown1
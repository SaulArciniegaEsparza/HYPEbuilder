"""
HYPEbuilder test

Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

#%% Set SAGA Environment
SAGA_PATH = r'D:\SAUL\DOCUMENTOS\PROGRAMAS\SAGA\saga-6.4.0_x64'

import os
import geopandas as gpd
from HYPEbuilder import gis_tools
from HYPEbuilder import GeoData

from pysaga import environment as env
env.set_env(SAGA_PATH)

PATH = '.'

#%% User variables
proj4 = '+proj=tmerc +lat_0=0 +lon_0=-84 +k=0.9999 +x_0=500000 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'

original_rasters = {
    'dem': os.path.join(PATH, 'data/srtm_20_11.tif'),
    'soil': os.path.join(PATH, 'data/TAXOUSDA_250m.tif'),
    'land': os.path.join(PATH, 'data/ESACCI_LandCover.tif')
}

output_folder = r'C:\Users\SAUL\Desktop\borrar'


#%% Transform grids
input_grids = {
    'dem': os.path.join(output_folder, 'dem.sgrd'),
    'soil': os.path.join(output_folder, 'soil.sgrd'),
    'land': os.path.join(output_folder, 'dem.sgrd'),
}


gis_tools.grid_transformation(
    input_grids['dem'],
    original_rasters['dem'],
    proj4,
    resampling=1
)
gis_tools.grid_transformation(
    input_grids['soil'],
    original_rasters['soil'],
    proj4,
    resampling=0
)

gis_tools.grid_transformation(
    input_grids['land'],
    original_rasters['land'],
    proj4,
    resampling=0
)

#%% Share grid extend and resolution

gis_tools.share_grid_extent(
    [input_grids['land'], input_grids['soil']],
    input_grids['land'],
    scale_up=0,
    scale_down=0
)


#%% Terrain preprocessing

terrain = gis_tools.dem_processing(
    output_folder,
    'terrain',
    input_grids['dem'],
    minslope=0.0001
)


#%% Streams and outlets

streams = gis_tools.channel_network(
    output_folder,
    'streams',
    terrain['dem'],
    terrain['flowdir'],
    terrain['flowacc'],
    init_value=1000,
    min_len=200
)


#%% Basins delimitation

basins = gis_tools.basins_delimitation(
        output_folder,
        'basins',
        streams['outlets'],
        terrain['dem'],
        terrain['flowdir'],
        terrain['slope'],
        streams['channels'],
        streams['streams'],
        proj4,
        find_regions=False
)


#%% Mask soil land using basins

gis_tools.grids_mask(
        [input_grids['land'], input_grids['soil']],
        basins['basins'],
        aspolygon=True
)


#%% Soil-Land combinatios

soil = gis_tools.soil_reclass(
        output_folder,
        'soil',
        input_grids['soil']
)

land = gis_tools.land_reclass(
        output_folder,
        'land',
        input_grids['land']
)

soil_land = gis_tools.soil_land_combinations(
        output_folder,
        'soil_land',
        basins['basins'],
        soil['soil'],
        land['land'],
        threshold=5
)



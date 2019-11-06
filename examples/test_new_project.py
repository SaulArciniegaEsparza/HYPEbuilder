# -*- coding: utf-8 -*-
"""
HYPEbuilder test

Create a new project


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

#%% Import modules needed
import os

from HYPEbuilder import Model
from HYPEbuilder import gis_tools
from HYPEbuilder import GeoData

from pysaga import environment as env


# Set SAGA GIS environment
ENV = r'C:\Users\zaula\Documents\PROGRAMAS\SAGA GIS\saga-6.4.0_x64'
env.set_env(ENV)

FILEPATH = os.path.dirname(os.path.abspath(__file__))


#%% Model parameters
FOLDER = os.path.join(FILEPATH, 'test_model')
FILES = {
        'geodata': os.path.join(FILEPATH, 'data', 'geodata.csv'),
        'geoclass': os.path.join(FILEPATH, 'data', 'geoclass.csv'),
        'geodataclass': os.path.join(FILEPATH, 'data', 'geodataclass.csv'),
}


#%% Create or open a new project
model = Model()

if os.path.exists(FOLDER):
    model.open_project(FOLDER)
else:
    model.new_project(
            FOLDER,
            FILES['geodata'],
            FILES['geoclass'],
            FILES['geodataclass'],
    )



# -*- coding: utf-8 -*-
"""
HYPEbuilder test

Open and run a model


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


#%% Open model
FOLDER = os.path.join(FILEPATH, 'test_model')

model = Model()
model.open_project(FOLDER)

#%% Add forcings
FILES = {
        'qobs': os.path.join(FILEPATH, 'data', 'Qobs.csv'),
        'pobs': os.path.join(FILEPATH, 'data', 'Pobs.csv'),
        'tobs': os.path.join(FILEPATH, 'data', 'Tobs.csv'),
}

model.Forcings.import_obs(**FILES)


#%% Define initial and final dates
model.Info['bdate'] = '1990-01-01'
model.Info['cdate'] = '1992-01-01'
model.Info['edate'] = '1995-12-31'
model.Info.write()


#%% Run Model
model.run_model()



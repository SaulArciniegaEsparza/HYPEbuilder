"""
HYPEbuilder test

Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

# Import modules needed
import os

from HYPEbuilder import HYPEmodel as model
from HYPEbuilder import gis_tools
from HYPEbuilder import GeoData

from pysaga import environment as env


# Set SAGA GIS environment
ENV = r'C:\Users\zaula\Documents\PROGRAMAS\SAGA GIS\saga-6.4.0_x64'
env.set_env(ENV)


# Set model folder
folder = r'C:\Users\zaula\Desktop\borrar\project1'

if os.path.exists(folder):
    model.open_project(folder)
else:
    model.new_project(folder, 'Project example', 5367)



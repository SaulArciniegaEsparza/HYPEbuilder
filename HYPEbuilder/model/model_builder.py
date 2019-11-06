"""
==============================================================================
HYPE hydrological model tools for python

This code defines the Model class, used as manager for the HYPE model
files creation and administration.


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
==============================================================================
"""

import sys as _sys
import os as _os
import subprocess as _sp

import numpy as _np
import pandas as _pd

from . import file_tools
from ..builder.geodata import GeoData
from . import time_series

from distutils.dir_util import copy_tree as _copy_tree

FILEPATH = _os.path.dirname(_os.path.abspath(__file__))

CLASS1 = type(_pd.Series())
CLASS2 = type(_pd.DataFrame())


# ==============================================================================
# Global parameters
# ==============================================================================

FOLDERS = ['results', 'forcings']


# ==============================================================================
# HYPE builder class
# ==============================================================================


class Model(object):
    """
    Class to create and run HYPE models
    """

    def __init__(self):
        self.path = None
        self.hype_path = None
        self.GeoData = None
        self.GeoClass = None
        self.Parameters = None
        self.Info = None
        self.Forcings = None
        self.BranchData = None
        self.Results = None

    def __repr__(self):
        return 'HYPEbuilder.Model'

    def new_project(self, folder, geodata, geoclass, geodataclass):
        """Create a new project with some default options"""
        if _os.path.exists(folder):
            raise FileExistsError('Project folder already exist!\nLoad project instead')

        # Create folders
        _os.makedirs(folder)
        self.path = folder
        self.check_folders()
        self.create_filedir()

        # Load default settings
        self.create_info()

        # Load geodata and geodataclass
        self.load_geodata(geodata)
        self.load_geodataclass(geodataclass)

        # Load geoclass
        self.load_geoclass(geoclass)

        # Create parameter templates
        self.create_parameters()

        # Add tools
        self.Forcings = time_series.Forcings(folder)
        self.Results = time_series.Results(folder)

        print(f'Project < {folder} > successfully created')

    def open_project(self, folder):
        """Open an existing folder project"""
        if _os.path.exists(folder):
            self.path = folder
            self.sync()
            self.Forcings = time_series.Forcings(folder)
            self.Results = time_series.Results(folder)
            print(f'Project < {folder} > successfully loaded')
        else:
            raise FileNotFoundError('Project folder does not exists')

    def load_geodata(self, geodata):
        """Load geodata information"""
        if self.path:
            self.GeoData = GeoData(geodata)
            filename = _os.path.join(self.path, 'GeoData.txt')
            self.GeoData.filename = filename
            self.GeoData.save_table(self.GeoData.filename)

    def load_geodataclass(self, geodataclass):
        """Load soil and land information for basins in GeoData"""
        if self.GeoData:
            self.GeoData.add_geodataclass(geodataclass)
            self.GeoData.save_table(self.GeoData.filename)

    def load_geoclass(self, geoclass):
        """Load geoclass information"""
        if self.path:
            filename = _os.path.join(self.path, 'GeoClass.txt')
            self.GeoClass = file_tools.FileGeoClass(filename)
            self.GeoClass.read(geoclass)
            self.GeoClass.write()

    def load_brachdata(self, brachdata):
        """Load branchdata information"""
        if self.path:
            filename = _os.path.join(self.path, 'BranchData.txt')
            self.BranchData = file_tools.FileBrachData(filename)
            self.BranchData.read(brachdata)
            self.BranchData.write()

    def partial_model_setup(self, subids=None, file=None, clean=False):
        """
        Create a submodel using subids as list or file.
        Also disables the submodel simulation.

        Parameters:
            subids     [list, tuple, array] input basins subids to create a submodel
            file       [string] input basins subids as file.
            clean      [bool] if True, subids and file are ignored. If True it removes
                        the pmsf.txt file and sets submodel as 'N'
        """
        if self.path:
            if clean:
                filename = _os.path.join(self.path, 'pmsf.txt')
                if _os.path.exists(filename):
                    _os.remove(filename)
                self.Info['submodel'] = 'N'
                self.Info.write()
                return

            if subids is None and type(file) is str:
                subids = _np.genfromtxt(file, dtype=int)

            if type(subids) in (list, tuple, _np.ndarray, CLASS1):
                if type(subids) is CLASS1:
                    subids = subids.values

                n = len(subids)

                if n == 0:
                    raise ValueError('Input subids is empty')

                subids = _np.array(subids, dtype=int)
                filename = _os.path.join(self.path, 'pmsf.txt')
                with open(filename, 'w') as fout:
                    fout.write(f'{n}\n')
                    for value in subids:
                        fout.write(f'{value}\t')

                self.Info['submodel'] = 'Y'
                self.Info.write()
            else:
                raise TypeError('Bad subids input type')

    def create_parameters(self):
        """Create a template of parameters"""
        if self.path:
            if self.GeoClass:
                filename = _os.path.join(self.path, 'par.txt')
                self.Parameters = file_tools.FilePar(filename)
                self.Parameters.load_template(self.GeoClass.count_land_class(),
                                              self.GeoClass.count_soil_class())
                self.Parameters.write()

    def create_info(self):
        """Create info.txt template"""

        if self.path:
            self.Info = file_tools.FileInfo(_os.path.join(self.path, 'info.txt'))
            self.Info.load_template()
            self.Info.write()

    def create_filedir(self):
        """Creates the filedir.txt for simulation"""

        filename = _os.path.join(self.path, 'filedir.txt')
        with open(filename, 'w') as fout:
            if _os.name == 'posix':
                fout.write("'./'")
            else:
                fout.write("'.\\'")

    def clean_results(self):
        """Clean results folder"""

        folder = _os.path.join(self.path, 'results')
        if _os.path.exists(folder):
            for file in _os.listdir(folder):
                _os.remove(_os.path.join(folder, file))

    def clean_log(self):
        """Deletes hyss... and test... files from model folder"""
        if self.path:
            for name in _os.listdir(self.path):
                if 'hyss' in name or 'tests' in name:
                    _os.remove(_os.path.join(self.path, name))

    def check_folders(self):
        """Create required folder if they are missing"""
        for subfolder in FOLDERS:
            folder = _os.path.join(self.path, subfolder)
            if not _os.path.exists(folder):
                _os.makedirs(folder)

    def build(self):
        """Save all the files loaded by HYPEbuilder class"""
        if self.path:
            if self.Info is not None:
                self.Info.write()
            if self.GeoData is not None:
                self.GeoData.save_table(self.GeoData.filename)
            if self.Parameters is not None:
                self.Parameters.write()
            if self.GeoClass is not None:
                self.GeoClass.write()
            if self.BranchData is not None:
                self.BranchData.write()

    def set_hype_path(self, path):
        """Set folder path where HYPE excecutable is located"""
        if _os.path.exists(path):
            self.hype_path = path

    def sync_info(self):
        """synchronizes data with the info.txt"""
        if self.path:
            filename = _os.path.join(self.path, 'info.txt')
            if _os.path.exists(filename):
                self.Info = file_tools.FileInfo(filename)
                self.Info.read()
            else:
                print('info.txt does not exist in folder!')

    def sync_parameters(self):
        """synchronizes data with the par.txt"""
        if self.path:
            filename = _os.path.join(self.path, 'par.txt')
            if _os.path.exists(filename):
                self.Parameters = file_tools.FilePar(filename)
                self.Parameters.read()
            else:
                print('par.txt does not exist in folder!')

    def sync_geodata(self):
        """synchronizes data with the GeoData.txt"""
        if self.path:
            filename = _os.path.join(self.path, 'GeoData.txt')
            if _os.path.exists(filename):
                self.GeoData = GeoData(filename)
                self.GeoData.filename = filename
            else:
                print('GeoData.txt does not exist in folder!')

    def sync_geoclass(self):
        """synchronizes data with the GeoClass.txt"""
        filename = _os.path.join(self.path, 'GeoClass.txt')
        if _os.path.exists(filename):
            self.GeoClass = file_tools.FileGeoClass(filename)
            self.GeoClass.read()
        else:
            print('GeoClass.txt does not exist in folder!')

    def sync_branchdata(self):
        """synchronizes data with the BranchData.txt"""
        filename = _os.path.join(self.path, 'BranchData.txt')
        self.BranchData = file_tools.FileBrachData(filename)
        if _os.path.exists(filename):
            self.BranchData.read()

    def sync(self):
        """Reload files info, geodata, geoclass and parameters"""
        self.sync_info()
        self.sync_geodata()
        self.sync_geoclass()
        self.sync_parameters()
        self.sync_branchdata()

    def export_results(self, out_folder):
        """Exports the results folder to other folder"""
        if self.path:
            in_folder = _os.path.join(self.path, 'results')
            if _os.path.exists(in_folder):
                if _os.path.exists(out_folder):
                    raise FileExistsError(f'Folder < {out_folder} > already exist!')
                else:
                    _copy_tree(in_folder, out_folder)

    def run_model(self, bdate=None, cdate=None, edate=None):
        """Run HYPE model"""
        if self.path is None:
            return 1
        if _os.name == 'posix':
            if self.hype_path is None:
                code = 'hype'
            else:
                code = _os.path.join(self.hype_path, 'hype')
        else:
            if self.hype_path is None:
                code = 'HYPE'
            else:
                code = _os.path.join(self.hype_path, 'HYPE')

        self.check_folders()
        self.clean_results()

        if type(bdate) is str:
            self.Info['bdate'] = bdate
        if type(cdate) is str:
            self.Info['cdate'] = cdate
        if type(edate) is str:
            self.Info['edate'] = edate
        if (type(bdate) is str or type(cdate) is str
                or type(edate) is str):
            self.Info.write()

        return _sp.call(code, cwd=self.path)


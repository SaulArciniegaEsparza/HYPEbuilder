"""
==============================================================================
HYPE hydrological model tools for python

File manager tools


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
==============================================================================
"""

import os
import re
import ast
import json
import toml
from copy import deepcopy

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

CLASS1 = type(pd.Series())
CLASS2 = type(pd.DataFrame())

FILEPATH = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# Load default parameters
# ==============================================================================

PAR_TEMPLATE = os.path.join(os.path.dirname(FILEPATH), 'templates',
                            'default_parameters.toml')

with open(PAR_TEMPLATE, 'r', encoding='utf-8-sig') as fid:
    PARAMETERS = toml.load(fid)


# ==============================================================================
# File info class
# ==============================================================================

class FileInfo(object):
    """info.txt manager"""

    def __init__(self, filename):
        self.filename = filename
        self.parameters = {}

    def __repr__(self):
        return 'hypeBUILDER.FileInfo'

    def __str__(self):
        return json.dumps(self.parameters, indent=4)

    def __getitem__(self, key):
        return self.parameters.get(key, None)

    def __setitem__(self, key, value):
        self.parameters[key] = value

    def __delitem__(self, key):
        del(self.parameters[key])

    def load_template(self):
        folder = os.path.dirname(FILEPATH)
        if os.name == 'posix':
            template = os.path.join(folder, 'templates', 'info_linux.txt')
        else:
            template = os.path.join(folder, 'templates', 'info_win.txt')
        self.read(template)

    def read(self, filename=None):
        """Read the input file or the file"""
        if filename is None:
            filename = self.filename

        if not os.path.exists(filename):
            print(f'File < {filename} > does not exist')
            return

        with open(filename, 'r', encoding='utf-8-sig') as fid:
            text = fid.readlines()

        # Read text as list of lists
        lines = []
        for line in text:
            line = line.rstrip()
            if not line.startswith('!!') and len(line) > 0:
                line = re.split('\t| ', re.sub(' +', ' ', line.replace("'", "")))

                if len(line) > 0:
                    new_values = []
                    for value in line:
                        if "-" not in value:
                            try:
                                new_values.append(ast.literal_eval(value))
                            except:
                                new_values.append(value)
                        else:
                            new_values.append(value)
                    lines.append(new_values)

        # List to nested dict
        parameters = {}
        list1 = deepcopy(lines)
        keys1 = np.array([str(x.pop(0)) for x in list1], dtype=object)
        for key1 in np.unique(keys1):
            nr1 = np.sum(key1 == keys1)
            if nr1 == 1:
                row1 = list1[np.where(keys1 == key1)[0][0]]
                ne1 = len(row1)
                if ne1 == 1:
                    parameters[key1] = row1[0]
                elif ne1 > 1:
                    parameters[key1] = row1
            else:
                parameters[key1] = {}
                list2 = [list1[i] for i in np.where(keys1 == key1)[0]]
                keys2 = np.array([str(x.pop(0)) for x in list2], dtype=object)
                for key2 in np.unique(keys2):
                    nr2 = np.sum(key2 == keys2)
                    if nr2 == 1:
                        row2 = list2[np.where(keys2 == key2)[0][0]]
                        ne2 = len(row2)
                        if ne2 == 1:
                            parameters[key1][key2] = row2[0]
                        elif ne2 > 1:
                            parameters[key1][key2] = row2
                    else:
                        parameters[key1][key2] = {}
                        list3 = [list2[i] for i in np.where(keys2 == key2)[0]]
                        keys3 = np.array([str(x.pop(0)) for x in list3], dtype=object)
                        for key3 in np.unique(keys3):
                            row3 = list3[np.where(keys3 == key3)[0][0]]
                            ne3 = len(row3)
                            if ne3 == 1:
                                parameters[key1][key2][key3] = row3[0]
                            elif ne3 > 1:
                                parameters[key1][key2][key3] = row3

        self.parameters.update(parameters)

    def sync(self):
        self.read()

    def write(self):
        with open(self.filename, 'w') as fout:
            for key, value in self.parameters.items():
                if type(value) in (int, float, str):
                    fout.write(f'{key}')
                    if key == 'modeldir':
                        if os.name == 'posix':
                            fout.write("\t'./'")
                        else:
                            fout.write("\t'.\\'")
                    elif key == 'resultdir':
                        if os.name == 'posix':
                            fout.write("\t'./results/'")
                        else:
                            fout.write("\t'.\\results\\'")
                    elif key == 'forcingdir':
                        if os.name == 'posix':
                            fout.write("\t'./forcings/'")
                        else:
                            fout.write("\t'.\\forcings\\'")
                    else:
                        fout.write(f'\t{value}')
                    fout.write('\n')
                elif type(value) in (list, tuple):
                    fout.write(f'{key}')
                    for val in value:
                        fout.write(f'\t{val}')
                    fout.write('\n')
                elif type(value) is dict:
                    for key1, value1 in value.items():
                        if type(value1) in (int, float, str):
                            fout.write(f'{key}\t{key1}')
                            fout.write(f'\t{value1}')
                            fout.write('\n')
                        elif type(value1) in (list, tuple):
                            fout.write(f'{key}\t{key1}')
                            for val in value1:
                                fout.write(f'\t{val}')
                            fout.write('\n')
                        elif type(value1) is dict:
                            for key2, value2 in value1.items():
                                if type(value2) in (int, float, str):
                                    fout.write(f'{key}\t{key1}\t{key2}')
                                    fout.write(f'\t{value2}')
                                    fout.write('\n')
                                elif type(value2) in (list, tuple):
                                    fout.write(f'{key}\t{key1}\t{key2}')
                                    for val in value2:
                                        fout.write(f'\t{val}')
                                    fout.write('\n')


# ==============================================================================
# Par class
# ==============================================================================

class FilePar(object):
    """par.txt file management"""

    def __init__(self, filename):
        self.filename = filename
        self.soil = None
        self.land = None
        self.general = None

    def __repr__(self):
        return 'hypeBUILDER.FilePar'

    def read(self):
        """Reads the par file"""

        if not os.path.exists(self.filename):
            print(f'File < {self.filename} > does not exist')
            return

        with open(self.filename, 'r', encoding='utf-8-sig') as fid:
            text = fid.readlines()

        params = dict(soil={}, land={}, general={})
        category = 'general'
        for line in text:
            line = line.strip()
            if line.startswith('!!'):
                if 'soil' in line.lower():
                    category = 'soil'
                elif 'land' in line.lower():
                    category = 'land'
                elif 'general' in line.lower():
                    category = 'general'
            elif not line.startswith('!!') and len(line) > 0:
                line = re.split('\t| |', re.sub(' +', ' ', line.replace("'", "")))
                key = line.pop(0)
                values = [float(x) for x in line]

                params[category][key] = values

        for key in params:
            if key in ('soil', 'land'):
                self.__dict__[key] = pd.DataFrame(params[key]).transpose()
                self.__dict__[key].columns = np.arange(1,
                                                       self.__dict__[key].shape[1]+1,
                                                       dtype=int)
            else:
                self.__dict__[key] = pd.DataFrame(params[key]).transpose().iloc[:, 0]

    def write(self):
        """Writes the parameters file par.txt"""

        with open(self.filename, 'w') as fout:
            for key in ('soil', 'land', 'general'):
                if self.__dict__[key] is not None:
                    if len(self.__dict__[key]) > 0:
                        fout.write(f'!! {key}\n')
                        if type(self.__dict__[key]) == CLASS1:
                            for index, val in self.__dict__[key].iteritems():
                                fout.write(f'{index}\t{val}\n')
                        else:
                            for index, row in self.__dict__[key].iterrows():
                                fout.write(f'{index}')
                                for val in row:
                                    fout.write(f'\t{val}')
                                fout.write('\n')

    def load_template(self, nland=2, nsoil=2):
        """Create parameter template file"""

        # Soil parameters
        soil_params = (list(PARAMETERS['soil']['runoff'].keys())
                       + list(PARAMETERS['soil']['water_content'].keys()))
        self.soil = pd.DataFrame(np.zeros((len(soil_params), int(nsoil))),
                                 columns=np.arange(1, nsoil+1, dtype=int),
                                 index=soil_params)
        # Land parameters
        land_params = (list(PARAMETERS['land']['pet'].keys())
                       + list(PARAMETERS['land']['surface_runoff'].keys()))
        self.land = pd.DataFrame(np.zeros((len(land_params), int(nland))),
                                 columns=np.arange(1, nland+1, dtype=int),
                                 index=land_params)
        # General parameters
        general_params = ['lp', 'cevpam', 'rivvel', 'rrcs3']
        self.general = pd.Series([1.0, 0.8, 0.8, 0.0001], index=general_params)

    def add_soil_parameter(self, key, value):
        """Add a soil parameter using a default value"""
        if type(value) in (int, float):
            self.soil.loc[key, :] = value
        elif len(value) == self.soil.shape[1]:
            self.soil.loc[key, :] = value

    def add_land_parameter(self, key, value):
        """Add a land parameter using a default value"""
        if type(value) in (int, float):
            self.land.loc[key, :] = value
        elif len(value) == self.land.shape[1]:
            self.land.loc[key, :] = value

    def add_general_parameter(self, key, value):
        """Add a general parameter using a default value"""
        if type(value) in (int, float):
            self.general.loc[key] = value

    def del_parameter(self, category, key):
        if category in self.__dict__:
            if key in self.__dict__[category]:
                self.__dict__[category].drop(key, axis=0, inplace=True)

    def load_soil_parameters(self, category):
        """Add the soil parameters from a category"""
        category = category.replace(' ', '_').lower()
        if category in PARAMETERS['soil']:
            for key, value in PARAMETERS['soil'][category].items():
                self.add_soil_parameter(key, value)

    def load_land_parameters(self, category):
        """Add the land parameters from a category"""
        category = category.replace(' ', '_').lower()
        if category in PARAMETERS['land']:
            for key, value in PARAMETERS['land'][category].items():
                self.add_soil_parameter(key, value)

    def load_geleral_parameters(self, category):
        """Add the general parameters from a category"""
        category = category.replace(' ', '_').lower()
        if category in PARAMETERS['general']:
            for key, value in PARAMETERS['general'][category].items():
                self.add_soil_parameter(key, value)

    def sync(self):
        """Reads the FilePar.txt"""
        self.read()

    @staticmethod
    def get_categories():
        """Returns a dictionary of categorical parameters"""
        def_params = dict.fromkeys(PARAMETERS.keys())
        for key in PARAMETERS:
            def_params[key] = list(PARAMETERS[key].keys())
        return def_params


# ==============================================================================
# GeoClass class
# ==============================================================================

class FileGeoclass(object):

    def __init__(self, filename):
        self.filename = filename
        self.data = None

    def read(self, filename=None):
        """Reads a GeoClass file"""
        if filename is None:
            filename = self.filename

        if os.path.exists(filename):
            if filename.endswith('.txt'):
                data = pd.read_csv(filename, delimiter='\t', encoding='utf-8-sig')
            else:
                data = pd.read_csv(filename, delimiter=',', encoding='utf-8-sig')

            self.data = data
        else:
            raise IOError(f'File < {filename} > does not exist')

    def write(self):
        """Writes the GeoClass.txt file"""

        if self.data is not None:
            data = self.data.copy()
            cols = list(data.columns)
            if '!!' not in cols[0]:
                cols[0] = f'!! {cols[0]}'
                data.columns = cols
            data.to_csv(self.filename, sep='\t', index=False)

    def count_soil_class(self):
        """Returns the number of unique soil classes"""
        if self.data is not None:
            nsoil = len(np.unique(self.data.iloc[:, 2].values))
            return nsoil

    def count_land_class(self):
        """Returns the number of unique land classes"""
        if self.data is not None:
            nland = len(np.unique(self.data.iloc[:, 1].values))
            return nland

    def sync(self):
        """Reads the GeoClass.txt"""
        self.read()

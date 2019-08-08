from setuptools import setup, find_packages

setup(
    name='HYPEbuilder',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/SaulArciniegaEsparza/HYPEbuilder',
    license='BSD',
    author='Saul Arciniega Esparza',
    author_email='zaul.ae@gmail.com',
    description='HYPE hydrological model builder using pysaga',
    include_package_data=True,
    # install_requires=[
    #     'numpy',
    #     'scipy',
    #     'pandas',
    #     'numba',
    #     'tables',
    #     'geopandas<0.5.0',
    #     'gdal<3.0.0',
    #     'pyshp==1.2.12',
    #     'toml',
    # ],
    classifiers=[
        'Development Status :: HYPE hydrological model builder using python',
        'Intended Audience :: Hydrology',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.6 or newer'
    ]
)

#!/usr/bin/env python
# ----------------------------------------------------------------------------
# GeoMed3d
# 
# This file is part of the GeoMed3D project: https://github.com/mobigroup/geomed3d
# 
# Copyright (c) 2023, Alexey Pechnikov
# 
# Licensed under the BSD 3-Clause License (see LICENSE for details)
# ----------------------------------------------------------------------------

from setuptools import setup, find_packages
import urllib.request

# read the contents of your README file
#from pathlib import Path
#this_directory = Path(__file__).parent
#long_description = (this_directory / "README.md").read_text()

upstream_url = 'https://raw.githubusercontent.com/mobigroup/geomed3d/main/README.md'
response = urllib.request.urlopen(upstream_url)
long_description = response.read().decode('utf-8')

setup(
    name='geomed3d',
    version='2023.12.03',
    description='GeoMed3D (Geologic Medium 3D) - 3D Density Inversion by Circular Hough Transform for Everyone',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mobigroup/geomed3d',
    author='Alexey Pechnikov',
    author_email='pechnikov@mobigroup.ru',
    license='BSD-3-Clause',
    packages=find_packages(),
    install_requires=['xarray>=0.19.0',
                      'numpy>=1.22.4',
                      'numba',
                      'dask',
                      'pandas>=1.5',
                      'geopandas',
                      'rasterio',
                      'rioxarray',
                      'scipy>=1.9.1',
                      'shapely>=2.0.1',
                      'rioxarray',
                      'h5netcdf>=1.2.0',
                      'h5py',
                      'earthengine-api',
                      'vtk'
                      ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.10'
    ],
    python_requires='>=3.10'
)

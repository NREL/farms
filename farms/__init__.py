# -*- coding: utf-8 -*-
"""
FARMS package

Notes
-----
cloud_types:
    "'N/A': -15,
    'Clear': 0,
    'Probably Clear': 1,
    'Fog': 2,
    'Water': 3,
    'Super-Cooled Water': 4,
    'Mixed': 5,
    'Opaque Ice': 6,
    'Cirrus': 7,
    'Overlapping': 8,
    'Overshooting': 9,
    'Unknown': 10,
    'Dust': 11,
    'Smoke': 12"
"""

import os

import pandas as pd

from .version import __version__

FARMSDIR = os.path.dirname(os.path.realpath(__file__))

RADIUS = pd.read_csv(
    os.path.join(FARMSDIR, "sun_earth_radius_vector.csv")
).set_index("doy")

# Constant clear/cloudy integer labels for use of AllSky
CLEAR_TYPES = (0, 1, 11, 12)
WATER_TYPES = (2, 3, 4, 5, 10)
ICE_TYPES = (6, 7, 8, 9)
CLOUD_TYPES = WATER_TYPES + ICE_TYPES

# Solar constant global variable. Flux density value measuring mean solar
# electromagnetic radiation per unit area, default is 1361.2 (W/m2).
SOLAR_CONSTANT = 1361.2

# Truncate irrad data when sza > this SZA limit
SZA_LIM = 89.0

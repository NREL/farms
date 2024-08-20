"""
PyTest file for FARMS-DNI

Created on 03/24/2023

@author: Yu Xie
"""

import numpy as np

from farms import farms_dni
from farms.utilities import execute_pytest


def test_farmsdni():
    """
    Test FARMS-DNI with typical input variables Check if the DNI computation is
    between 0 and 1400, and larger than the DNI in the narrow beam.
    """
    n = 100
    F0 = np.full(n, 1360.0)
    tau = np.arange(n) * 0.1
    sza = np.arange(n) * 0.85
    solar_zenith_angle = np.cos(sza * np.pi / 180.0)
    De = np.arange(n) + 10.0
    phase = np.full(n, 2)
    phase[:50] = 1
    Tddclr = (np.arange(n) + 100) * 0.005
    Ftotal = np.flip(np.arange(n) + 900.0)
    F1 = Ftotal * 0.7

    Fd, dni_farmsdni, dni0 = farms_dni.farms_dni(
        F0,
        tau,
        solar_zenith_angle,
        De,
        phase,
        Tddclr,
        Ftotal,
        F1,
    )

    cond1 = dni_farmsdni[dni_farmsdni < 0]
    cond2 = dni_farmsdni[dni_farmsdni >= 1400]
    cond3 = np.where(dni_farmsdni < dni0)[0]

    print(Fd)  # Set this to pass the Linting

    wrong_num = len(cond1) + len(cond2) + len(cond3)
    assert wrong_num == 0


if __name__ == "__main__":
    execute_pytest(__file__)

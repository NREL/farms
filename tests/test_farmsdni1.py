"""
PyTest file for FARMS-DNI

Created on 03/24/2023

@author: Yu Xie
"""

import numpy as np
from farms import farms_dni


def test_farmsdni():
    '''Test FARMS-DNI with typical input variables '''

    n = 100
    F0 = np.full(n, 1360.0)
    tau = np.arange(n) * 0.1
    sza = np.arange(n) * 0.85
    solar_zenith_angle = np.cos(sza * np.pi / 180.0)
    De = np.arange(n) + 10.0
    phase = np.full(n, 2)
    phase[:50] = 1
    phase1 = np.where(phase == 1)[0]
    phase2 = np.where(phase == 2)[0]
    Tddclr = (np.arange(n) + 100) * 0.005
    Ftotal = np.flip(np.arange(n) + 900.0)
    F1 = Ftotal * 0.7

    Fd, dni_farmsdni, dni0 = farms_dni.farms_dni(
        F0, tau, solar_zenith_angle, De, phase, phase1, phase2,
        Tddclr, Ftotal, F1
    )

    cond1 = dni_farmsdni[dni_farmsdni < 0]
    cond2 = dni_farmsdni[dni_farmsdni >= 1400]
    cond3 = np.where(dni_farmsdni < dni0)[0]

    Fd = 0.0 #Set this to pass the Linting

    wrong_num = len(cond1) + len(cond2) + len(cond3)
    assert wrong_num == 0


if __name__ == "__main__":
    test_farmsdni()

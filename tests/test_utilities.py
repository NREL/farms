# pylint: skip-file
"""
PyTest file for farms utilities.

Created on June 7th 2019.

@author: gbuster
"""

import numpy as np

from farms.utilities import execute_pytest, rayleigh

RTOL = 0.001
ATOL = 0.001


def test_rayleigh():
    """Test the rayleigh violation check."""

    dhi = np.array(
        [
            [0.72444256, 0.23359043],
            [0.81759012, 0.88451122],
            [0.6008738, 0.03514198],
            [0.95301128, 0.36230728],
            [0.65396809, 0.25557211],
        ]
    )
    cs_dhi = np.array(
        [
            [0.82444256, 0.23359043],
            [0.81759012, 0.88451122],
            [0.7008738, 0.03514198],
            [0.95301128, 0.36230728],
            [0.65396809, 0.35557211],
        ]
    )

    fill_out = np.array(
        [[1, 0], [0, 0], [1, 0], [0, 0], [0, 1]], dtype=np.int16
    )

    fill_flag = np.zeros_like(dhi).astype(np.int16)

    fill_flag = rayleigh(dhi, cs_dhi, fill_flag, rayleigh_flag=1)

    assert np.array_equal(fill_flag, fill_out)


if __name__ == "__main__":
    execute_pytest(__file__)

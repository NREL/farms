"""
Created on Fri June 1 2015
FAST Model
Adapted from Yu Xie IDL Fast Radiative Transfer Model
@author: Anthony Lopez

This Fast All-sky Radiation Model for Solar applications (FARMS) was developed
by Yu Xie (Yu.Xie@nrel.gov). Please contact him for more information.

Literature
----------
[1] Yu Xie, Manajit Sengupta, Jimy Dudhia, "A Fast All-sky Radiation Model
    for Solar applications (FARMS): Algorithm and performance evaluation",
    Solar Energy, Volume 135, 2016, Pages 435-445, ISSN 0038-092X,
    https://doi.org/10.1016/j.solener.2016.06.003.
    (http://www.sciencedirect.com/science/article/pii/S0038092X16301827)
"""
import collections
import numpy as np

from farms import CLEAR_TYPES, ICE_TYPES, WATER_TYPES, SOLAR_CONSTANT
import farms.utilities as ut


def water_phase(tau, De, solar_zenith_angle):
    """Get cloudy Tducld and Ruucld for the water phase."""
    # 12a from [1]
    Ptau = (2.8850 + 0.002 * (De - 60.0)) * solar_zenith_angle - 0.007347

    # 12b from [1]
    PDHI = (0.7846 * (1.0 + 0.0002 * (De - 60.0))
            * np.power(solar_zenith_angle, 0.1605))

    # 12c from [1]
    delta = (-0.644531 * solar_zenith_angle + 1.20117 + 0.129807
             / solar_zenith_angle - 0.00121096
             / (solar_zenith_angle * solar_zenith_angle) + 1.52587e-07
             / (solar_zenith_angle * solar_zenith_angle * solar_zenith_angle))

    # part of 12d from [1]
    y = 0.012 * (tau - Ptau) * solar_zenith_angle

    # 11 from [1]
    Tducld = ((1.0 + np.sinh(y)) * PDHI
              * np.exp(-(np.power(np.log10(tau) - np.log10(Ptau), 2.0))
              / delta))

    # 14a from [1]
    Ruucld = np.where(tau < 1.0,
                      0.107359 * tau,
                      1.03 - np.exp(-(0.5 + np.log10(tau))
                                    * (0.5 + np.log10(tau)) / 3.105))

    return Tducld, Ruucld


def ice_phase(tau, De, solar_zenith_angle):
    """Get cloudy Tducld and Ruucld for the ice phase."""
    # 13a from [1]
    Ptau = np.where(De <= 26.0,
                    2.8487 * solar_zenith_angle - 0.0029,
                    ((2.8355 + (100.0 - De) * 0.006) * solar_zenith_angle
                     - 0.00612))

    # 13b from [1]
    PDHI = 0.756 * np.power(solar_zenith_angle, 0.0883)

    # 13c from [1]
    delta = (-0.0549531 * solar_zenith_angle + 0.617632
             + (0.17876 / solar_zenith_angle)
             - (0.002174 / solar_zenith_angle ** 2))

    # part of 13c from [1]
    y = 0.01 * (tau - Ptau) * solar_zenith_angle

    # 11 from [1]
    Tducld = ((1.0 + np.sinh(y)) * PDHI
              * np.exp(-(np.power(np.log10(tau) - np.log10(Ptau), 2.0))
                       / delta))

    # 14b from [1]
    Ruucld = np.where(tau < 1.0,
                      0.094039 * tau,
                      1.02 - np.exp(-(0.5 + np.log10(tau))
                                    * (0.5 + np.log10(tau)) / 3.25))

    return Tducld, Ruucld


def farms(tau, cloud_type, cloud_effective_radius, solar_zenith_angle,
          radius, Tuuclr, Ruuclr, Tddclr, Tduclr, albedo, debug=False):
    """Fast All-sky Radiation Model for Solar applications (FARMS).

    Literature
    ----------
    [1] Yu Xie, Manajit Sengupta, Jimy Dudhia, "A Fast All-sky Radiation Model
        for Solar applications (FARMS): Algorithm and performance evaluation",
        Solar Energy, Volume 135, 2016, Pages 435-445, ISSN 0038-092X,
        https://doi.org/10.1016/j.solener.2016.06.003.
        (http://www.sciencedirect.com/science/article/pii/S0038092X16301827)

    Variables
    ---------
    F0
        Radiative flux at top of atmosphere
    Fd
        Direct solar flux in the downwelling direction at the surface
        (eq 2a from [1])
    De
        Effective cloud particle size (diameter).


    Parameters
    ----------
    tau : np.ndarray
        Cloud optical thickness (cld_opd_dcomp) (unitless).
    cloud_type : np.ndarray
        Integer values representing different cloud types
        https://github.nrel.gov/PXS/pxs/wiki/Cloud-Classification
    cloud_effective_radius : np.ndarray
        Cloud effective particle radius (cld_reff_dcomp) (micron).
    solar_zenith_angle : np.ndarray
        Solar zenith angle (degrees). Must represent the average value over the
        integration period (e.g. hourly) under scrutiny.
    radius : np.ndarray
        Sun-earth radius vector, varies between 1.017 in July and
        0.983 in January.
    Tuuclr : np.ndarray
        Transmittance of the clear-sky atmosphere for diffuse incident and
        diffuse outgoing fluxes (uu).
        ***Calculated from multiple REST2 runs at different solar angles.
        Average of Tddclr w different solar angles (see eq 5 from [1]).
    Ruuclr : np.ndarray
        Calculated in REST2. Aerosol reflectance for diffuse fluxes.
    Tddclr : np.ndarray
        Calculated in REST2. Transmittance of the clear-sky atmosphere for
        direct incident and direct outgoing fluxes (dd).
        Tddclr = dni / etdirn
    Tduclr : np.ndarray
        Calculated in REST2. Transmittance of the clear-sky atmosphere for
        direct incident and diffuse outgoing fluxes (du).
        Tduclr = dhi / (etdirn * cosz)
    albedo : np.ndarray
        Ground albedo.
    debug : bool
        Flag to output additional transmission/reflectance variables.

    Returns
    -------
    ghi : np.ndarray
        FARMS GHI values (this is the only output if debug is False).
    fast_data : collections.namedtuple
        Additional debugging variables if debug is True.
        Named tuple with irradiance data. Attributes:
            ghi : global horizontal irradiance (w/m2)
            dni : direct normal irradiance (w/m2)
            dhi : diffuse horizontal irradiance (w/m2)
    """
    # disable divide by zero warnings
    np.seterr(divide='ignore')

    ut.check_range(Tddclr, 'Tddclr')
    ut.check_range(Tduclr, 'Tduclr')
    ut.check_range(Ruuclr, 'Ruuclr')
    ut.check_range(Tuuclr, 'Tuuclr')
    ut.check_range(tau, 'tau (cld_opd_dcomp)', rang=(0, 160))

    F0 = SOLAR_CONSTANT / (radius * radius)
    solar_zenith_angle = np.cos(np.radians(solar_zenith_angle))

    phase = np.zeros_like(cloud_type)
    phase[np.in1d(cloud_type, WATER_TYPES).reshape(cloud_type.shape)] = 1
    phase[np.in1d(cloud_type, ICE_TYPES).reshape(cloud_type.shape)] = 2

    phase1 = np.where(phase == 1)
    phase2 = np.where(phase == 2)

    De = 2.0 * cloud_effective_radius

    Tducld = np.zeros_like(tau)
    Ruucld = np.zeros_like(tau)

    Tducld[phase1], Ruucld[phase1] = water_phase(tau[phase1], De[phase1],
                                                 solar_zenith_angle[phase1])

    Tducld[phase2], Ruucld[phase2] = ice_phase(tau[phase2], De[phase2],
                                               solar_zenith_angle[phase2])

    # eq 8 from [1]
    Tddcld = np.exp(-tau / solar_zenith_angle)

    Fd = solar_zenith_angle * F0 * Tddcld * Tddclr  # eq 2a from [1]
    F1 = solar_zenith_angle * F0 * (Tddcld * (Tddclr + Tduclr)
                                    + Tducld * Tuuclr)  # eq 3 from [1]

    # ghi eqn 6 from [1]
    ghi = F1 / (1.0 - albedo * (Ruuclr + Ruucld * Tuuclr * Tuuclr))
    dni = Fd / solar_zenith_angle  # eq 2b from [1]
    dhi = ghi - Fd  # eq 7 from [1]

    clear_mask = np.in1d(cloud_type, CLEAR_TYPES).reshape(cloud_type.shape)
    if debug:
        # Return NaN if clear-sky, else return cloudy sky data
        fast_data = collections.namedtuple('fast_data', ['ghi', 'dni', 'dhi',
                                                         'Tddcld', 'Tducld',
                                                         'Ruucld'])
        fast_data.Tddcld = np.where(clear_mask, np.nan, Tddcld)
        fast_data.Tducld = np.where(clear_mask, np.nan, Tducld)
        fast_data.Ruucld = np.where(clear_mask, np.nan, Ruucld)
        fast_data.ghi = np.where(clear_mask, np.nan, ghi)
        fast_data.dni = np.where(clear_mask, np.nan, dni)
        fast_data.dhi = np.where(clear_mask, np.nan, dhi)

        return fast_data
    else:
        # return only GHI
        return np.where(clear_mask, np.nan, ghi)

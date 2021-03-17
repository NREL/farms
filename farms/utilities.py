"""Common utilities FARMS module.
"""

from copy import deepcopy
import pandas as pd
import numpy as np
import os
from warnings import warn
from farms import RADIUS, CLEAR_TYPES, CLOUD_TYPES, SZA_LIM


def check_range(data, name, rang=(0, 1)):
    """Ensure that data values are in correct range."""
    if np.nanmin(data) < rang[0] or np.nanmax(data) > rang[1]:
        raise ValueError('Variable "{n}" is out of expected '
                         'transmittance/reflectance range. Recommend checking '
                         'solar zenith angle to ensure cos(sza) is '
                         'non-negative and non-zero. '
                         'Max/min of {n} = {mx}/{mn}'
                         .format(n=name,
                                 mx=np.nanmax(data),
                                 mn=np.nanmin(data)))


def ti_to_radius_csv(time_index, n_cols=1):
    """Convert a time index to radius.

    Parameters
    ----------
    time_index : pandas.core.indexes.datetimes.DatetimeIndex
        NSRDB time series. Can extract from h5 as follows:
        time_index = pd.to_datetime(h5['time_index'][...].astype(str))
    n_cols : int
        Number of columns to output. The radius vertical 1D array will be
        copied this number of times horizontally (np.tile).

    Returns
    -------
    radius : np.array
        Array of radius values matching the time index.
        Shape is (len(time_index), n_cols).
    """
    doy = pd.DataFrame(index=time_index.dayofyear)
    radius = doy.join(RADIUS)
    radius = np.tile(radius.values, n_cols)

    return radius


def ti_to_radius(time_index, n_cols=1):
    """Calculates Earth-Sun Radius Vector.

    Reference:
    http://www.nrel.gov/docs/fy08osti/34302.pdf

    Parameters
    ----------
    time_index : pandas.core.indexes.datetimes.DatetimeIndex
        NSRDB time series. Can extract from h5 as follows:
        time_index = pd.to_datetime(h5['time_index'][...].astype(str))
    n_cols : int
        Number of columns to output. The radius vertical 1D array will be
        copied this number of times horizontally (np.tile).

    Returns
    -------
    radius : np.array
        Array of radius values matching the time index.
        Shape is (len(time_index), n_cols).
    """
    # load earth periodic table
    path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(path, 'earth_periodic_terms.csv'))
    df['key'] = 1
    # 3.1.1 (4). Julian Date.
    j = time_index.to_julian_date().values
    # 3.1.2 (5). Julian Ephermeris Date
    j = j + 64.797 / 86400
    # 3.1.3 (7). Julian Century Ephemeris
    j = (j - 2451545) / 36525
    # 3.1.4 (8). Julian Ephemeris Millennium
    j = j / 10
    df_jme = pd.DataFrame({'uid': range(len(j)), 'jme': j, 'key': 1})
    # Merge JME with Periodic Table
    df_merge = pd.merge(df_jme, df, on='key')
    # 3.2.1 (9). Heliocentric radius vector.
    df_merge['r'] = df_merge['a'] * np.cos(df_merge['b'] + df_merge['c']
                                           * df_merge['jme'])
    # 3.2.2 (10).
    dfs = df_merge.groupby(by=['uid', 'term'])['r'].sum().unstack()
    # 3.2.4 (11). Earth Heliocentric radius vector
    radius = ((dfs['R0'] + dfs['R1'] * j + dfs['R2'] * np.power(j, 2)
               + dfs['R3'] * np.power(j, 3) + dfs['R4'] * np.power(j, 4)
               + dfs['R5'] * np.power(j, 5)) / np.power(10, 8)).values
    radius = radius.reshape((len(time_index), 1))
    radius = np.tile(radius, n_cols)

    return radius


def calc_beta(aod, alpha):
    """Calculate the Angstrom turbidity coeff. (beta).

    Parameters
    ----------
    aod : np.ndarray
        Array of aerosol optical depth (AOD) values. Shape must match alpha.
    alpha : np.ndarray
        Array of angstrom wavelength exponent values. Shape must match aod.

    Returns
    -------
    beta : np.ndarray
        Array of Angstrom turbidity coeff., i.e. AOD at 1000 nm.
        Shape will be same as aod and alpha. Will be tested for compliance
        with the mandatory interval [0, 2.2].
    """
    if aod.shape != alpha.shape:
        raise ValueError('To calculate beta, aod and alpha inputs must be of '
                         'the same shape. Received arrays of shape {} and {}'
                         .format(aod.shape, alpha.shape))

    beta = aod * np.power(0.55, alpha)
    if np.max(beta) > 2.2 or np.min(beta) < 0:
        warn('Calculation of beta resulted in values outside of '
             'expected range [0, 2.2]. Min/max of beta are: {}/{}'
             .format(np.min(beta), np.max(beta)))

    return beta


def calc_dhi(dni, ghi, sza):
    """Calculate the diffuse horizontal irradiance and correct the direct.

    Parameters
    ----------
    dni : np.ndarray
        Direct normal irradiance.
    ghi : np.ndarray
        Global horizontal irradiance.
    sza : np.ndarray
        Solar zenith angle (degrees).

    Returns
    -------
    dhi : np.ndarray
        Diffuse horizontal irradiance. This is ensured to be non-negative.
    dni : np.ndarray
        Direct normal irradiance. This is set to zero where dhi < 0
    """
    dhi = ghi - dni * np.cos(np.radians(sza))
    if np.min(dhi) < 0:
        # patch for negative DHI values. Set DNI to zero, set DHI to GHI
        pos = np.where(dhi < 0)
        dni[pos] = 0
        dhi[pos] = ghi[pos]

    return dhi, dni


def rayleigh(dhi, cs_dhi, fill_flag, rayleigh_flag=7):
    """Perform Rayleigh violation check (all-sky diffuse >= clearsky diffuse).

    Decided not to use this in all-sky on 7/3/2019

    Failed data gets filled with farms data

    Parameters
    ----------
    dhi : np.ndarray
        All-sky diffuse irradiance.
    cs_dhi : np.ndarray
        Clearsky (rest) diffuse irradiance.
    fill_flag : np.ndarray
        Array of integers signifying whether irradiance has been filled.
    rayleigh_flag : int
        Fill flag for rayleigh violation.

    Returns
    -------
    fill_flag : np.ndarray
        Array of integers signifying whether irradiance has been filled, with
        rayleigh violations marked with the rayleigh flag.
    """
    # boolean mask where the rayleigh check fails
    # (compare agains 99.9% to avoid false positives)
    failed = (dhi < (0.999 * deepcopy(cs_dhi))) & (cs_dhi > 0)
    fill_flag[failed] = rayleigh_flag

    return fill_flag


def merge_rest_farms(clearsky_irrad, cloudy_irrad, cloud_type):
    """Combine clearsky and rest data into all-sky irradiance array.

    Parameters
    ----------
    clearsky_irrad : np.ndarray
        Clearsky irradiance data from REST.
    cloudy_irrad : np.ndarray
        Cloudy irradiance data from FARMS.
    cloud_type : np.ndarray
        Cloud type array which acts as a mask specifying where to take
        cloud/clear data.

    Returns
    -------
    all_sky_irrad : np.ndarray
        All-sky (cloudy + clear) irradiance data, merged dataset from
        FARMS and REST.
    """
    # disable nan warnings
    np.seterr(divide='ignore', invalid='ignore')

    # combine clearsky and farms according to the cloud types.
    all_sky_irrad = np.where(np.isin(cloud_type, CLEAR_TYPES),
                             clearsky_irrad, cloudy_irrad)

    return all_sky_irrad


def screen_cld(cld_data, rng=(0, 160)):
    """Enforce a numeric range on cloud property data.

    Parameters
    ----------
    cld_data : np.ndarray
        Cloud property data (cld_opd_dcomp, cld_reff_dcomp).
    rng : list | tuple
        Inclusive intended range of the cloud data.

    Parameters
    ----------
    cld_data : np.ndarray
        Cloud property data (cld_opd_dcomp, cld_reff_dcomp)
        with min/max values equal to rng.
    """
    cld_data[np.isnan(cld_data)] = 0
    cld_data[(cld_data < rng[0])] = rng[0]
    cld_data[(cld_data > rng[1])] = rng[1]

    return cld_data


def screen_sza(sza, lim=SZA_LIM):
    """Enforce an upper limit on the solar zenith angle.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle in degrees.
    lim : int | float
        Upper limit of SZA in degrees.

    Returns
    ----------
    sza : np.ndarray
        Solar zenith angle in degrees with max value = lim.
    """
    sza[(sza > lim)] = lim

    return sza


def dark_night(irrad_data, sza, lim=SZA_LIM):
    """Enforce zero irradiance when solar angle >= threshold.

    Parameters
    ----------
    irrad_data : np.ndarray
        DHI, DNI, or GHI.
    sza : np.ndarray
        Solar zenith angle in degrees.
    lim : int | float
        Upper limit of SZA in degrees.

    Returns
    -------
    irrad_data : np.ndarray
        DHI, DNI, or GHI with zero values when sza >= lim.
    """
    night_mask = np.where(sza >= lim)
    irrad_data[night_mask] = 0

    return irrad_data


def cloud_variability(irrad, cs_irrad, cloud_type, var_frac=0.05,
                      distribution='uniform', option='tri', tri_center=0.9,
                      random_seed=123):
    """Add syntehtic variability to irradiance when it's cloudy.

    Parameters
    ----------
    irrad : np.ndarray
        Full FARMS + REST2 merged irradiance 2D array.
    cs_irrad : np.ndarray
        REST2 clearsky irradiance without bad or missing data.
    cloud_type : np.ndarray
        Array of numerical cloud types.
    var_frac : float
        Maximum variability fraction (0.05 is 5% variability) or if
        distribution is "normal" this is the maximum relative standard
        deviation of the Variability.
    distribution : str
        Distribution shape of the Variability. Can be uniform or normal.
    option : str
        Variability function option ('tri' or 'linear').
    random_seed : int | NoneType
        Number to seed the numpy random number generator. Used to generate
        reproducable psuedo-random cloud variability. Numpy random will be
        seeded with the system time if this is None.

    Returns
    -------
    irrad : np.ndarray
        Full FARMS + REST2 merged irradiance 2D array with variability added
        to cloudy timesteps.
    """
    # disable divide by zero warnings
    np.seterr(divide='ignore', invalid='ignore')

    if var_frac:
        # set a seed for psuedo-random but repeatable results
        np.random.seed(seed=random_seed)

        # update the clearsky ratio (1 is clear, 0 is cloudy or dark)
        csr = irrad / cs_irrad
        # Set the cloud/clear ratio to zero when it's nighttime
        csr[(cs_irrad == 0)] = 0

        if distribution == 'uniform':
            variability_scalar = uniform_variability(csr, cloud_type, var_frac,
                                                     option=option,
                                                     tri_center=tri_center)
        elif distribution == 'normal':
            variability_scalar = normal_variability(csr, cloud_type, var_frac,
                                                    option=option,
                                                    tri_center=tri_center)
        else:
            raise ValueError('Did not recognize distribution: {}'
                             .format(distribution))

        irrad *= variability_scalar

    return irrad


def uniform_variability(csr, cloud_type, var_frac, option='tri',
                        tri_center=0.9):
    """Get an array with uniform variability scalars centered at 1 that can be
    multiplied by a irradiance array with the same shape as csr.

    Parameters
    ----------
    csr : np.ndarray
        REST2 clearsky irradiance without bad or missing data.
        This is a 2D array with (time, sites).
    cloud_type : np.ndarray
        Array of numerical cloud types.
    var_frac : float
        Maximum variability fraction (0.05 is 5% variability).
    option : str
        Variability function option ('tri' or 'linear').
    tri_center : float
        Value of the clearsky ratio at which there is maximum variability
        (only used for the triangular distribution).

    Returns
    -------
    variability_scalar : np.ndarray
        Array with shape matching csr with uniform random numbers centered at
        1 with range (1 - var_frac) to (1 + var_frac). This array can be
        multiplied by an irradiance array with the same shape as csr
    """
    if option == 'linear':
        var_frac_arr = linear_variability(csr, var_frac)
    elif option == 'tri':
        var_frac_arr = tri_variability(csr, var_frac, tri_center=tri_center)
    else:
        raise ValueError('Did not recognize variability option: {}'
                         .format(option))

    # get a uniform random scalar array 0 to 1 with data shape
    rand_arr = np.random.rand(csr.shape[0], csr.shape[1])

    # Center the random array at 1 +/- var_frac_arr (with csr scaling)
    variability_scalar = 1 + var_frac_arr * (rand_arr * 2 - 1)

    # only apply rand to the applicable cloudy timesteps
    variability_scalar = np.where(np.isin(cloud_type, CLOUD_TYPES),
                                  variability_scalar, 1)

    return variability_scalar


def normal_variability(csr, cloud_type, var_frac, option='tri',
                       tri_center=0.9):
    """Get an array with a normal distribution of variability scalars centered
    at 1 that can be multiplied by a irradiance array with the same shape as
    csr.

    Parameters
    ----------
    csr : np.ndarray
        REST2 clearsky irradiance without bad or missing data.
        This is a 2D array with (time, sites).
    cloud_type : np.ndarray
        Array of numerical cloud types.
    var_frac : float
        One relative standard deviation variability (0.05 is a relative
        standard deviation of 5% variability).
    option : str
        Variability function option ('tri' or 'linear').
    tri_center : float
        Value of the clearsky ratio at which there is maximum variability
        (only used for the triangular distribution).

    Returns
    -------
    variability_scalar : np.ndarray
        Array with shape matching csr with normally distributed random numbers
        centered at 1 with range (1 - var_frac) to (1 + var_frac). This array
        can be multiplied by an irradiance array with the same shape as csr
    """
    if option == 'linear':
        var_frac_arr = linear_variability(csr, var_frac)
    elif option == 'tri':
        var_frac_arr = tri_variability(csr, var_frac, tri_center=tri_center)
    else:
        raise ValueError('Did not recognize variability option: {}'
                         .format(option))

    # get a normal distribution of data centered at 0 with stdev 1
    rand_arr = np.random.normal(loc=0.0, scale=1.0, size=csr.shape)

    # Center the random array at 1 +/- var_frac_arr (with csr scaling)
    variability_scalar = 1 + var_frac_arr * rand_arr

    # only apply rand to the applicable cloudy timesteps
    variability_scalar = np.where(np.isin(cloud_type, CLOUD_TYPES),
                                  variability_scalar, 1)

    return variability_scalar


def linear_variability(csr, var_frac):
    """Return an array with a linear relation between clearsky ratio and
    maximum variability fraction. Each value in the array is the maximum
    variability fraction for the corresponding clearsky ratio.

    Parameters
    ----------
    csr : np.ndarray
        REST2 clearsky irradiance without bad or missing data.
        This is a 2D array with (time, sites).
    var_frac : float
        Maximum variability fraction (0.05 is 5% variability).

    Returns
    -------
    out : np.ndarray
        Array with shape matching csr with maximum variability (var_frac)
        when the csr = 1 (clear or thin clouds). Each value in the array is
        the maximum variability fraction for the corresponding clearsky ratio.

    """
    return var_frac * csr


def tri_variability(csr, var_frac, tri_center=0.9):
    """Return an array with a triangular distribution between clearsky ratio
    and maximum variability fraction. Each value in the array is
    the maximum variability fraction for the corresponding clearsky ratio.


    The max variability occurs when csr==tri_center, and zero variability when
    csr==0 or csr==1

    Parameters
    ----------
    csr : np.ndarray
        REST2 clearsky irradiance without bad or missing data.
        This is a 2D array with (time, sites).
    var_frac : float
        Maximum variability fraction (0.05 is 5% variability).
    tri_center : float
        Value of the clearsky ratio at which there is maximum variability.

    Returns
    -------
    tri : np.ndarray
        Array with shape matching csr with maximum variability (var_frac)
        when the csr==tri_center. Each value in the array is the maximum
        variability fraction for the corresponding clearsky ratio.
    """
    tri_left = var_frac * csr * 1.11111
    slope = -1 / (1 - tri_center)
    yint = tri_center * 10 + 1
    tri_right = var_frac * (slope * csr + yint)
    tri = np.where(csr < tri_center, tri_left, tri_right)

    return tri

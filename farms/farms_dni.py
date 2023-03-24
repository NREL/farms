"""
Created on March 1, 2022
Upgraded on March 21, 2023
FARMS-DNI model developed by Yu Xie (yu.xie@nrel.gov)

References
-----------
Xie, Y., Sengupta, M., Dudhia, J., 2016. A Fast All-sky Radiation Model for
Solar applications (FARMS): Algorithm and performance evaluation. Sol. Energy
 135, 435-445.
Xie, Y., Sengupta, M., Liu, Y., Long, H., Min, Q., Liu, W., Habte, A., 2020.
A physics-based DNI model assessing all-sky circumsolar radiation. iScience 22,
 doi.org/10.1016/j.isci.2020.100893.
Yang, J., Xie, Y., Sengupta, M., Liu, Y., Long, H., 2022. Parameterization of
cloud transmittance for expeditious assessment and forecasting of all-sky DNI.
J. Renewable Sustainable Energy 14, 063703.
"""

import numpy as np
import sys
import pandas as pd


def TDD2(Z, Ftotal, F1):
    '''
    Compute surface reflection that falls in the circumsolar region.
    A parameterization of Eq. (S4) in Xie et al (2020) is used.
    '''
    # compute the integration of the solid angle
    a = 5.94991536e-03
    b = 5.42116600e-01
    c = 331280.9859904468
    muomega = a * np.exp(-np.power(Z - b, 3.0) / c)

    Fd2 = np.cos(Z * np.pi / 180.0) * (Ftotal - F1) * muomega / np.pi
    return Fd2


def TDDP(Z, tau, De, phase1, phase2):
    '''
    compute cloud transmittance of DNI for water and ice clouds
    '''
    Tddcld = np.zeros_like(tau)
    Tddcld[phase1] = Pwater(Z[phase1], tau[phase1], De[phase1])
    Tddcld[phase2] = Pice(Z[phase2], tau[phase2], De[phase2])

    return Tddcld


def Pwater(Z, tau, De):
    '''
    Compute cloud transmittance for water clouds
    '''
    umu0 = np.cos(Z * np.pi / 180.0)
    # taup  Eq.(3) in Yang et al. (2022)
    taup = np.zeros_like(Z)
    taup[(De < 10.0) & (umu0 < 0.1391)] = 0.1
    taup[(De < 10.0) & (umu0 >= 0.1391) & (umu0 < 0.2419)] = 0.2
    taup[(De < 10.0) & (umu0 >= 0.2419) & (umu0 < 0.3090)] = 0.3
    taup[(De < 10.0) & (umu0 >= 0.3090) & (umu0 < 0.4067)] = 0.4
    taup[(De < 10.0) & (umu0 >= 0.4067) & (umu0 < 0.6156)] = 0.5
    taup[(De < 10.0) & (umu0 >= 0.6156)] = 1.0

    taup[(De >= 10.0) & (umu0 < 0.1391)] = 0.1
    taup[(De >= 10.0) & (umu0 >= 0.1391) & (umu0 < 0.2079)] = 0.2
    taup[(De >= 10.0) & (umu0 >= 0.2079) & (umu0 < 0.3090)] = 0.3
    taup[(De >= 10.0) & (umu0 >= 0.3090) & (umu0 < 0.3746)] = 0.4
    taup[(De >= 10.0) & (umu0 >= 0.3746) & (umu0 < 0.6156)] = 0.5
    taup[(De >= 10.0) & (umu0 >= 0.6156)] = 1.0

    # Tddp  Eq(4) in Yang et al. (2022)
    h = 0.005553 * np.log(De) + 0.002503
    h[De == 0] = 0.0
    Tddp = np.zeros_like(Z)
    a1 = (umu0 >= 0.0) & (umu0 < 0.342)
    Tddp[a1] = h[a1] * \
        (-0.1787 * umu0[a1] * umu0[a1] + 0.2207 * umu0[a1] + 0.977)
    a2 = (umu0 >= 0.342) & (umu0 < 0.4694)
    Tddp[a2] = h[a2]
    a3 = (umu0 >= 0.4694) & (umu0 < 0.7193)
    Tddp[a3] = h[a3] * \
        (2.6399 * umu0[a3] * umu0[a3] - 3.2111 * umu0[a3] + 1.9434)
    a4 = (umu0 >= 0.7193) & (umu0 < 0.8829)
    Tddp[a4] = h[a4] * \
        (-0.224 * umu0[a4] * umu0[a4] + 0.0835 * umu0[a4] + 1.056)
    a5 = (umu0 >= 0.8829) & (umu0 < 0.9396)
    Tddp[a5] = h[a5] * \
        (-94.381 * umu0[a5] * umu0[a5] + 170.32 * umu0[a5] - 75.843)
    a6 = (umu0 >= 0.9396) & (umu0 < 0.9945)
    Tddp[a6] = h[a6] * \
        (-12.794 * umu0[a6] * umu0[a6] + 22.686 * umu0[a6] - 8.9392)
    a7 = (umu0 >= 0.9945) & (umu0 < 0.999)
    Tddp[a7] = h[a7] * \
        (11248.61 * umu0[a7] * umu0[a7] - 22441.07 * umu0[a7] + 11193.59)
    a8 = umu0 >= 0.999
    Tddp[a8] = 0.76 * h[a8]

    # Eq.(6) in Yang et al. (2022)
    a = 2.0339 * np.power( umu0, -0.927 )
    b = 6.6421 * np.power( umu0, 2.0672 )

    # compute Tddcld using Eq.(5) in Yang et al. (2022)
    Tddcld = np.zeros_like(Z)
    a1 = tau <= 0.9 * taup
    Tddcld[a1] = Tddp[a1] * np.tanh(a[a1] * tau[a1])
    a2 = (tau > 0.9 * taup) & (tau < taup)
    Tddcld[a2] = Tddp[a2] * np.tanh(0.9 * a[a2] * taup[a2]) + Tddp[a2] * \
    (np.tanh(b[a2] / np.power(taup[a2],2.0)) - np.tanh(0.9 * a[a2] * taup[a2]))\
        * (tau[a2] - 0.9 * taup[a2])/(0.1 * taup[a2])
    a3 = tau >= taup
    Tddcld[a3] = Tddp[a3] * np.tanh(b[a3] / np.power(tau[a3],2.0))

    return Tddcld


def Pice(Z, tau, De):
    '''
    Compute cloud transmittance for ice clouds
    '''
    umu0 = np.cos( Z * np.pi / 180.0 )
    # taup Eq.(3) in Yang et al. (2022)
    taup = np.zeros_like(Z)
    a1 = (De >= 5.0) & (De < 14.0) & (umu0 < 0.1391)
    taup[a1] = 0.1
    a2 = (De >= 5.0) & (De < 14.0) & (umu0 >= 0.1391) & (umu0 < 0.2079)
    taup[a2] = 0.2
    a3 = (De >= 5.0) & (De < 14.0) & (umu0 >= 0.2079) & (umu0 < 0.3090)
    taup[a3] = 0.3
    a4 = (De >= 5.0) & (De < 14.0) & (umu0 >= 0.3090) & (umu0 < 0.3746)
    taup[a4] = 0.4
    a5 = (De >= 5.0) & (De < 14.0) & (umu0 >= 0.3746) & (umu0 < 0.6156)
    taup[a5] = 0.5
    a6 = (De >= 5.0) & (De < 14.0) & (umu0 >= 0.6156) & (umu0 < 0.9994)
    taup[a6] = 1.0
    a7 = (De >= 5.0) & (De < 14.0) & (umu0 >= 0.9994)
    taup[a7] = 1.5

    a8 = (De >= 14.0) & (De < 50.0) & (umu0 < 0.139173)
    taup[a8] = 0.1
    a9 = (De >= 14.0) & (De < 50.0) & (umu0 >= 0.139173) & \
         (umu0 < -0.0011 * De + 0.2307)
    taup[a9] = 0.2
    a10 = (De >= 14.0) & (De < 50.0) & (umu0 >= -0.0011 * De + 0.2307) & \
          (umu0 < -0.0022 * De + 0.3340)
    taup[a10] = 0.3
    a11 = (De >= 14.0) & (De < 50.0) & (umu0 >= -0.0022 * De + 0.3340) & \
          (umu0<-0.0020*De+0.4096)
    taup[a11] = 0.4
    a12 = (De >= 14.0) & (De < 50.0) & (umu0 >= -0.0020 * De + 0.4096) & \
          (umu0 < -0.0033 * De + 0.6461)
    taup[a12] = 0.5
    a13 = (De >= 14.0) & (De < 50.0) & (umu0 >= -0.0033 * De + 0.6461) & \
          (umu0 < -0.0049 * De + 1.0713)
    taup[a13] = 1.0
    a14 = (De >= 14.0) & (De < 50.0) & (umu0 >= -0.0049 * De + 1.0713)
    taup[a14] = 1.5
    a15 = (De >= 50.0) & (umu0 < -0.0006 * De + 0.2109)
    taup[a15] = 0.2
    a16 = (De >= 50.0) & (umu0 >= -0.0006 * De + 0.2109) & \
          (umu0 < -0.0005 * De + 0.2581)
    taup[a16] = 0.3
    a17 = (De >= 50.0) & (umu0 >= -0.0005 * De + 0.2581) & \
          (umu0 < -0.0010 * De + 0.3907)
    taup[a17] = 0.4
    a18 = (De >= 50.0) & (umu0 >= -0.0010 * De + 0.3907) & \
          (umu0 < -0.0008 * De + 0.4900)
    taup[a18] = 0.5
    a19 = (De >= 50.0) & (umu0 >= -0.0008 * De + 0.4900) & \
          (umu0 < -0.0017 * De + 0.8708)
    taup[a19] = 1.0
    a20 = (De >= 50.0) & (umu0 >= -0.0017 * De + 0.8708) & \
          (umu0 < -0.0006 * De + 1.0367)
    taup[a20] = 1.5
    a21 = (De >= 50.0) & (umu0 >= -0.0006 * De + 1.0367)
    taup[a21] = 2.0

    # Tddp   Eq(4) in Yang et al. (2022)
    Tddp = np.zeros_like(Z)
    b1 = (umu0 >= 0.9994) & (De <= 10.0)
    Tddp[b1] = 0.12269
    b2 = (umu0 >= 0.9994) & (De > 10.0) & (De <= 16.0)
    Tddp[b2] = 0.0015 * De[b2] + 0.1078
    b3 = (umu0 >= 0.9994) & (De > 16.0)
    Tddp[b3] = 0.1621 * np.exp(-0.016 * De[b3])

    b4 = (umu0 < 0.9396) & (De <= 10.0)
    Tddp[b4] = 0.14991
    b5 = (umu0 < 0.9945) & (umu0 >= 0.9396) & (De <= 10.0)
    Tddp[b5] = -4.5171 * np.power(umu0[b5],2.0) + 8.3056 * umu0[b5] - 3.6476
    b6 = (umu0 >= 0.9945) & (umu0 < 0.9994) & (De <= 10.0)
    Tddp[b6] = 298.45 * np.power(umu0[b6],2.0) - 601.33 * umu0[b6] + 303.04

    ade = -0.000232338 * np.power(De,2.0) + 0.012748726 * De + 0.046745083
    b7 = (umu0 < 0.999) & (De > 10.0) & (De <= 30.0) & (umu0 <= 0.2419)
    Tddp[b7] = (-8.454 * np.power(umu0[b7],2.0) + 2.4095 * umu0[b7] + 0.8425) *\
                ade[b7]
    b8 = (umu0 < 0.999) & (De > 10.0) & (De <= 30.0) & (umu0 > 0.2419) &\
         (umu0 <= 0.3746)
    Tddp[b8] = (-13.528 * np.power(umu0[b8],2.0) + 7.8403 * umu0[b8] -0.1221)\
               * ade[b8]
    b9 = (umu0 < 0.999) & (De > 10.0) & (De <= 30.0) & (umu0 > 0.3746) &\
         (umu0 <= 0.4694)
    Tddp[b9] = (19.524 * np.power(umu0[b9],2.0) - 16.5 * umu0[b9] + 4.4612) *\
                ade[b9]
    b10 = (umu0 < 0.999) & (De > 10.0) & (De <= 30.0) & (umu0 > 0.4694) &\
          (umu0 <= 0.5877)
    Tddp[b10] = (16.737 * np.power(umu0[b10],2.0) - 17.419 * umu0[b10] + 5.4881)\
                 * ade[b10]
    b11 = (umu0 < 0.999) & (De > 10.0) & (De <= 30.0) & (umu0 > 0.5877) &\
          (umu0 <= 0.6691)
    Tddp[b11] = (-39.493 * np.power(umu0[b11],2.0) + 48.963 * umu0[b11] -14.175)\
                * ade[b11]
    b12 = (umu0 < 0.999) & (De > 10.0) & (De <= 30.0) & (umu0 > 0.6691) &\
          (umu0 <= 0.7660)
    Tddp[b12] = (0.4017 * np.power(umu0[b12],2.0) - 0.243 * umu0[b12] + 0.9609)\
                *ade[b12]
    b13 = (umu0 < 0.999) & (De > 10.0) & (De <= 30.0) & (umu0 > 0.7660) &\
          (umu0 <= 0.8480)
    Tddp[b13] = (-11.183 * np.power(umu0[b13],2.0) + 18.126 * umu0[b13] - \
                 6.3417)*ade[b13]
    b14 = (umu0 < 0.999) & (De > 10.0) & (De <= 30.0) & (umu0 > 0.8480) & \
          (umu0 <= 0.8987)
    Tddp[b14] = (-163.36 * np.power(umu0[b14],2.0) + 283.35 * umu0[b14] \
                 - 121.91) * ade[b14]
    b15 = (umu0 < 0.999) & (De > 10.0) & (De <= 30.0) & (umu0 > 0.8987) \
           & (umu0 <= 0.9396)
    Tddp[b15] = (-202.72 * np.power(umu0[b15],2.0) + 368.75 * umu0[b15] \
                 - 166.75) * ade[b15]
    b16 = (umu0 < 0.999) & (De > 10.0) & (De <= 30.0) & (umu0 > 0.9396) \
           & (umu0 <= 0.9702)
    Tddp[b16] = (-181.72 * np.power(umu0[b16],2.0) + 343.59 * umu0[b16] \
                 - 161.3)*ade[b16]
    b17 = (umu0 < 0.999) & (De > 10.0) & (De <= 30.0) & (umu0 > 0.9702) \
          & (umu0 <= 0.9945)
    Tddp[b17] = (127.66 * np.power(umu0[b17],2.0) - 255.73 * umu0[b17] \
                 + 129.03) * ade[b17]
    b18 = (umu0 < 0.999) & (De > 10.0) & (De <= 30.0) & (umu0 > 0.9945)
    Tddp[b18] = (908.66 * np.power(umu0[b18],2.0) - 1869.3 * umu0[b18] \
                 + 961.63) * ade[b18]

    bde = 0.0000166112 * np.power(De,2.0) - 0.00410998 * De + 0.352026619
    b19 = (umu0 < 0.999) & (De > 30.0) & (umu0 <= 0.2419)
    Tddp[b19] = (-4.362 * np.power(umu0[b19],2.0) - 0.0878 * umu0[b19] \
                 + 1.1218) * bde[b19]
    b20 = (umu0 < 0.999) & (De > 30.0) & (umu0 > 0.2419) & (umu0 <= 0.3746)
    Tddp[b20] = (-49.566 * np.power(umu0[b20],2.0) + 28.767 * umu0[b20] \
                 - 3.1299) * bde[b20]
    b21 = (umu0 < 0.999) & (De > 30.0) & (umu0 > 0.3746) & (umu0 <= 0.4694)
    Tddp[b21] = (58.572 * np.power(umu0[b21],2.0) - 49.5 * umu0[b21] \
                 + 11.363) * bde[b21]
    b22 = (umu0 < 0.999) & (De > 30.0) & (umu0 > 0.4694) & (umu0 <= 0.5877)
    Tddp[b22] = (62.118 * np.power(umu0[b22],2.0) - 63.037 * umu0[b22]
                 + 16.875) * bde[b22]
    b23 = (umu0 < 0.999) & (De > 30.0) & (umu0 > 0.5877) & (umu0 <= 0.6691)
    Tddp[b23] = (-237.68 * np.power(umu0[b23],2.0) + 293.21 * umu0[b23]
                 - 89.328) * bde[b23]
    b24 = (umu0 < 0.999) & (De > 30.0) & (umu0 > 0.6691) & (umu0 <= 0.7660)
    Tddp[b24] = (1.2051 * np.power(umu0[b24],2.0) - 0.7291 * umu0[b24]
                 + 0.8826) * bde[b24]
    b25 = (umu0 < 0.999) & (De > 30.0) & (umu0 > 0.7660) & (umu0 <= 0.8480)
    Tddp[b25] = (-55.6 * np.power(umu0[b25],2.0) + 90.698 * umu0[b25]
                 - 35.905) * bde[b25]
    b26 = (umu0 < 0.999) & (De > 30.0) & (umu0 > 0.8480) & (umu0 <= 0.8987)
    Tddp[b26] = (-422.36 * np.power(umu0[b26],2.0) + 733.97 * umu0[b26]
                 - 317.89) * bde[b26]
    b27 = (umu0 < 0.999) & (De > 30.0) & (umu0 > 0.8987) & (umu0 <= 0.9396)
    Tddp[b27] = (-457.09 * np.power(umu0[b27],2.0) + 831.11 * umu0[b27]
                 - 376.85) * bde[b27]
    b28 = (umu0 < 0.999) & (De > 30.0) & (umu0 > 0.9396) & (umu0 <= 0.9702)
    Tddp[b28] = (-344.91 * np.power(umu0[b28],2.0) + 655.67 * umu0[b28]
                 -310.5) * bde[b28]
    b29 = (umu0 < 0.999) & (De > 30.0) & (umu0 > 0.9702) & (umu0 <= 0.9945)
    Tddp[b29] = (622.85*np.power(umu0[b29],2.0) - 1227.6*umu0[b29]
                 +605.97)*bde[b29]
    b30 = (umu0 < 0.999) & (De > 30.0) & (umu0 > 0.9945)
    Tddp[b30] = (6309.63 * np.power(umu0[b30],2.0) - 12654.78 * umu0[b30]
                 + 6346.15) * bde[b30]

###compute Tddcld using Eq.(5) in Yang et al. (2022)
    a = 1.7686 * np.power(umu0,-0.95)
    b = 7.117 * np.power(umu0,1.9658)

    Tddcld = np.zeros_like(Z)
    a1 = tau <= 0.9 * taup
    Tddcld[a1] = Tddp[a1] * np.tanh(a[a1] * tau[a1])
    a2 = (tau > 0.9 * taup) & (tau < taup)
    Tddcld[a2] = Tddp[a2] * np.tanh(0.9 * a[a2] * taup[a2]) + \
                 Tddp[a2] * (np.tanh(b[a2] / np.power(taup[a2],2.0)) \
                  - np.tanh(0.9 * a[a2] * taup[a2])) * \
                 (tau[a2] - 0.9 * taup[a2]) / (0.1 * taup[a2])
    a3 = tau >= taup
    Tddcld[a3] = Tddp[a3] * np.tanh(b[a3] / np.power(tau[a3],2.0))

    return Tddcld


def farms_dni(F0, tau, solar_zenith_angle, De, phase, phase1, phase2, Tddclr, Ftotal, F1):
    '''
    Fast All-sky Radiation Model for solar applications with direct normal
    irradiance (FARMS-DNI)

    References
    -----------
    Xie, Y., Sengupta, M., Dudhia, J., 2016. A Fast All-sky Radiation Model for
    Solar applications (FARMS): Algorithm and performance evaluation. Sol. Energy
     135, 435-445.
    Xie, Y., Sengupta, M., Liu, Y., Long, H., Min, Q., Liu, W., Habte, A., 2020.
    A physics-based DNI model assessing all-sky circumsolar radiation. iScience 22,
     doi.org/10.1016/j.isci.2020.100893.
    Yang, J., Xie, Y., Sengupta, M., Liu, Y., Long, H., 2022. Parameterization of
    cloud transmittance for expeditious assessment and forecasting of all-sky DNI.
    J. Renewable Sustainable Energy 14, 063703.

    Input data
    -----------
    F0:  np.ndarray
        extraterrestrial solar radiation (Wm-2).
    tau : np.ndarray
        Cloud optical thickness (cld_opd_dcomp) (unitless).
    solar_zenith_angle : np.ndarray
        Solar zenith angle (degrees). Must represent the average value over the
        integration period (e.g. hourly) under scrutiny.
    De: np.ndarray
        Effective cloud particle size (diameter, micron).
    phase: np.ndarray
        Cloud thermodynamic phase (water:1, ice:2)
    phase1: np.ndarray
        np.where( phase==1 )
    phase2: np.ndarray
        np.where( phase==2 )
    Tddclr : np.ndarray
        Calculated in REST2. Transmittance of the clear-sky atmosphere for
        direct incident and direct outgoing fluxes (dd).
        Tddclr = dni / etdirn
    Ftotal: np.ndarray
        GHI (Wm-2)
    F1: np.ndarray
        First order solar radiation given in FARMS (Wm-2). See Xie et al. (2016)
        for more details.

    Output data
    -----------
    Fd: np.ndarray
        Direct radiation in the downwelling direction (Wm-2). It includes the
        narrow beam and the scattered radiation in the circumsolar region.
    dni_farmsdni: np.ndarray
        DNI computed by FARMS-DNI (Wm-2).
    dni0: np.ndarray
        DNI computed by the Lambert law (Wm-2). It only includes the narrow beam
         in the circumsolar region.
    '''

    # scale tau for the computation of DNI. See Eqs. (3a and 3b) in Xie et al. (2020), iScience.
    taudni = np.zeros_like(tau)

    a1 = np.where( (phase == 1) & (tau < 8.0) )
    a2 = np.where( (phase == 1) & (tau >= 8.0) )
    taudni[a1] = ( 0.254825 * tau[a1] - 0.00232717 * np.power(tau[a1], 2.0)  \
             + (5.19320e-06) * np.power(tau[a1], 3.0) ) * \
               (1.0 + (8.0 - tau[a1]) * 0.07)
    taudni[a2] = 0.2 * np.power(tau[a2] - 8.0, 1.5) + 2.10871

    b1 = np.where( (phase == 2) & (tau < 8.0) )
    b2 = np.where( (phase == 2) & (tau >= 8.0) )
    taudni[b1] = 0.345353 * tau[b1] - 0.00244671 * np.power(tau[b1], 2.0)  \
             + (4.74263E-06) * np.power(tau[b1], 3.0)
    taudni[b2] = 0.2 * np.power(tau[b2] - 8.0, 1.5) + 2.91345

    # compute DNI in the narrow beam. Eq.(S2) in Xie et al. (2020), iScience.
    Z = np.arccos(solar_zenith_angle) * 180.0 / np.pi
    dni0 = F0 * Tddclr * np.exp(-tau / solar_zenith_angle)

    Tddcld0 = np.exp(-taudni / solar_zenith_angle)
    Fd0 = solar_zenith_angle * F0 * Tddcld0 * Tddclr

    # compute scattered radiation in the circumsolar region. Eq.(S3 and S4) in Xie et al. (2020), iScience.
    Tddcld1 = TDDP(Z, taudni, De, phase1, phase2)
    Fd1 = solar_zenith_angle * F0 * Tddclr * Tddcld1
    Fd2 = TDD2(Z, Ftotal, F1 )

    # compute DNI using the three components. Eq.(S1) in Xie et al. (2020), iScience.
    Fd = Fd0 + Fd1 + Fd2
    dni_farmsdni = Fd / solar_zenith_angle

    return Fd, dni_farmsdni, dni0





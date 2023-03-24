
import numpy as np
from farms import farms_dni
import sys


def test_farmsdni():

    n = 100
    F0 = np.arange(n)
    F0[:] = 1360.0
    tau = np.arange(n)*0.1
    sza = np.arange(n)*0.85
    solar_zenith_angle = np.cos( sza*np.pi/180.0 )
    De = np.arange(n)+10.0
    phase = np.arange(n)
    phase[:50] = 1
    phase[50:] = 2
    phase1 = np.where(phase == 1)
    phase2 = np.where(phase == 2)
    Tddclr = ( np.arange(n)+100 )*0.005
    Ftotal = np.flip( np.arange(n) + 900.0)
    F1 = Ftotal*0.7 

    Fd, dni_farmsdni, dni0 = farms_dni.farms_dni(F0, tau, solar_zenith_angle, De, phase, phase1, phase2, Tddclr, Ftotal, F1)

    cond1 = [x for x in dni_farmsdni if x<0]
    cond2 = [x for x in dni_farmsdni if x>=1400]    
    cond3 = [i for i in range(dni_farmsdni.shape[0]) if dni_farmsdni[i] < dni0[i] ]

    wrong_num = np.size(cond1+cond2+cond3)
    assert wrong_num == 0



if __name__ == '__main__':
    execute_pytest()


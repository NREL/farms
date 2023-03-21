
import numpy as np
from farms import farms_dni
import sys

def sunearth(juday):
    b = 2.0*np.pi*juday/365.0
    R1 =  1.00011 + 0.034221*np.cos(b) + 0.001280*np.sin(b) + 0.000719*np.cos(2.0*b) +0.000077*np.sin(2.0*b)
    R = np.power(R1, -0.5)
    return R


f = open( '/projects/pxs/yxie/forecast/FARMS-DNI/test/ppdata/SRRL/datawater/input.dat', 'r')
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

print( dni_farmsdni)
print(  dni0 )

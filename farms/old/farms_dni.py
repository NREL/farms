"""
Created on March 1, 2022
FARMS-DNI model developed by Yu Xie (yu.xie@nrel.gov)
Must have Interomega.dat to compute DNI

Literature
[1] Yu Xie, Manajit Sengupta, Yangang Liu, Hai Long, Qilong Min, Weijia Liu, Aron Habte, 
    A physics-based DNI model assessing all-sky circumsolar radiation, iScience
[2] Yu Xie, Jaemo Yang, Manajit Sengupta, Yangang Liu, Xin Zhou, 
    Improving the prediction of DNI with physics-based representation of all-sky circumsolar radaition, Solar Energy
[3] Jaemo Yang, Yu Xie, Manajit Sengupta, Yangang Liu, Hai Long, 
    Parameterization of cloud transmittance for expeditious assessment and forecasting of all-sky DNI, Solar Energy

"""
import numpy as np
import sys
import pandas as pd


def Read_Two_Column_File(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))

    return x, y

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx



def TDD2(Z, Ftotal, Fd1 ):

    theta0, mmuomega = Read_Two_Column_File('Interomega.dat')
    theta00, idx = find_nearest(theta0, Z)
    muomega = mmuomega[idx]
    Fd2 = (Ftotal - Fd1)*muomega/np.pi
#    print( theta00, idx, muomega )
    return Fd2


def TDDP(Z, tau, De, phase):

#    Tddcld[phase1] = pwater( Z[phase1],tau[phase1], De[phase1] )
#    Tddcld[phase2] = pice( Z[phase1],tau[phase1], De[phase1] )

    return Tddcld



def Pwater(Z, tau, De):

    umu0 = np.cos( Z*np.pi/180.0 )
### taup
    taup = np.zeros_like(Z)
    taup[umu0<0.1391] = 0.1
    taup[(umu0>=0.1391) & (umu0<0.2419)] = 0.2
    taup[(umu0>=0.2419) & (umu0<0.3090)] = 0.3
    taup[(umu0>=0.3090) & (umu0<0.4067)] = 0.4
    taup[(umu0>=0.4067) & (umu0<0.6156)] = 0.5
    taup[umu0>=0.6156] = 1.0

###Tddp
    Tddp = np.zeros_like(Z)
    a1 = (umu0>=0.0) & (umu0<0.45)
    Tddp[a1] = 0.00628502*( np.log(De[a1]) + 0.035 )
    a2 = (umu0>=0.45) & (umu0<0.88)
    Tddp[a2] = 0.00609272*( np.log(De[a2]) + 0.035 )
    a3 = (umu0>=0.88) & (umu0<0.92)
    Tddp[a3] = ( -0.0128*umu0[a3] + 0.0175 )*( np.log(De[a3]) + 0.035 )
    a4 = (umu0>=0.92) & (umu0<0.99)
    Tddp[a4] = ( -0.0103*umu0[a4] + 0.0163 )*( np.log(De[a4]) + 0.035 )
    a5 = (umu0>=0.99) & (umu0<0.999)
    Tddp[a5] = ( -0.0326*umu0[a5] + 0.0392 )*( np.log(De[a5]) + 0.035 )
    a6 = umu0>=0.999
    Tddp[a6] = 0.0042*np.log(De[a6]) + 0.002

    a = 2.0461*np.power( umu0, -0.816 )
    b = 6.5257*np.power( umu0, 1.819 )

    Tddcld = np.zeros_like(Z)
    a1 = tau<=0.9*taup
    Tddcld[a1] = Tddp[a1]*np.tanh(a[a1]*tau[a1])
    a2 = (tau>0.9*taup) & (tau<taup)
    Tddcld[a2] = Tddp[a2]*np.tanh(0.9*a[a2]*taup[a2]) + Tddp[a2]*(np.tanh(b[a2]/np.power(taup[a2],2.0)) \
                  - np.tanh(0.9*a[a2]*taup[a2]))*(tau[a2]-0.9*taup[a2])/(0.1*taup[a2]) 
    a3 = tau>=taup
    Tddcld[a3] = Tddp[a3]*np.tanh(b[a3]/np.power(tau[a3],2.0))

    return Tddcld




def Pice(Z, tau, De):

    umu0 = np.cos( Z*np.pi/180.0 )
### taup
    taup = np.zeros_like(Z)
    a1 = (De>=5.0) & (De<14.0) & (umu0<0.1045)
    taup[a1] = 0.1
    a2 = (De>=5.0) & (De<14.0) & (umu0>=0.1045) & (umu0<0.1736)
    taup[a2] = 0.2
    a3 = (De>=5.0) & (De<14.0) & (umu0>=0.1736) & (umu0<0.2756)
    taup[a3] = 0.3
    a4 = (De>=5.0) & (De<14.0) & (umu0>=0.2756) & (umu0<0.3420)
    taup[a4] = 0.4
    a5 = (De>=5.0) & (De<14.0) & (umu0>=0.3420) & (umu0<0.5877)
    taup[a5] = 0.5
    a6 = (De>=5.0) & (De<14.0) & (umu0>=0.5877) & (umu0<0.9993)
    taup[a6] = 1.0
    a7 = (De>=5.0) & (De<14.0) & (umu0>=0.9993)
    taup[a7] = 1.5

    a8 = (De>=14.0) & (De<50.0) & (umu0<0.10453)
    taup[a8] = 0.1
    a9 = (De>=14.0) & (De<50.0) & (umu0>=0.10453) & (umu0<-0.0011*De+0.1966)
    taup[a9] = 0.2
    a10 = (De>=14.0) & (De<50.0) & (umu0>=-0.0011*De+0.1966) & (umu0<-0.0022*De+0.3009)
    taup[a10] = 0.3
    a11 = (De>=14.0) & (De<50.0) & (umu0>=-0.0022*De+0.3009) & (umu0<-0.0021*De+0.3774)
    taup[a11] = 0.4
    a12 = (De>=14.0) & (De<50.0) & (umu0>=-0.0021*De+0.3774) & (umu0<-0.0034*De+0.6188)
    taup[a12] = 0.5
    a13 = (De>=14.0) & (De<50.0) & (umu0>=-0.0034*De+0.6188) & (umu0<-0.0054*De+1.0743)
    taup[a13] = 1.0
    a14 = (De>=14.0) & (De<50.0) & (umu0>=-0.0054*De+1.0743)
    taup[a14] = 1.5

    a15 = (De>=50.0) & (umu0<-0.0006*De+0.1766)
    taup[a15] = 0.2
    a16 = (De>=50.0) & (umu0>=-0.0006*De+0.1766) & (umu0<-0.0005*De+0.2242)
    taup[a16] = 0.3
    a17 = (De>=50.0) & (umu0>=-0.0005*De+0.2242) & (umu0<-0.0011*De+0.3583)
    taup[a17] = 0.4
    a18 = (De>=50.0) & (umu0>=-0.0011*De+0.3583) & (umu0<-0.0008*De+0.4592)
    taup[a18] = 0.5
    a19 = (De>=50.0) & (umu0>=-0.0008*De+0.4592) & (umu0<-0.0018*De+0.8521)
    taup[a19] = 1.0
    a20 = (De>=50.0) & (umu0>=-0.0018*De+0.8521) & (umu0<-0.0007*De+1.0455)
    taup[a20] = 1.5
    a21 = (De>=50.0) & (umu0>=-0.0007*De+1.0455) 
    taup[a21] = 2.0


    Tddp = np.zeros_like(Z)
    b1 = (umu0>=0.999) & (De<=10.0) 
    Tddp[b1] = 0.12269
    b2 = (umu0>=0.999) & (De>10.0) & (De<=16.0)
    Tddp[b2] = 0.0015*De[b2] + 0.1078
    b3 = (umu0>=0.999) & (De>16.0)
    Tddp[b3] = 0.1621*np.exp(-0.016*De[b3])

    b4 = (umu0<0.9271) & (De<=10.0)
    Tddp[b4] = 0.1499
    b5 = (umu0<0.9902) & (umu0>=0.9271) & (De<=10.0)
    Tddp[b5] = -3.1476*np.power(De[b5],2.0) + 5.6543*De[b5] - 2.3682
    b6 = (umu0>=0.9902) & (umu0<0.999) & (De<=10.0)
    Tddp[b6] = 226.5*np.power(De[b6],2.0) - 454.41*De[b6] - 228.07


    ade = -0.000232338*np.power(De,2.0) + 0.012749*De + 0.046745
    b7 = (umu0<0.999) & (De>10.0) & (De<=30.0) & (umu0<=0.2079)
    Tddp[b7] = (-8.3642*np.power(umu0[b7],2.0) + 1.8051*umu0[b7] +0.9168)*ade[b7]
    b8 = (umu0<0.999) & (De>10.0) & (De<=30.0) & (umu0>0.2079) & (umu0<=0.3420)
    Tddp[b8] = (-13.253*np.power(umu0[b8],2.0) + 6.7924*umu0[b8] +0.1436)*ade[b8]
    b9 = (umu0<0.999) & (De>10.0) & (De<=30.0) & (umu0>0.3420) & (umu0<=0.4383)
    Tddp[b9] = (18.976*np.power(umu0[b9],2.0) - 14.82*umu0[b9] +3.87)*ade[b9]
    b10 = (umu0<0.999) & (De>10.0) & (De<=30.0) & (umu0>0.4383) & (umu0<=0.5591)
    Tddp[b10] = (16.078*np.power(umu0[b10],2.0) - 15.767*umu0[b10] +4.8211)*ade[b10]
    b11 = (umu0<0.999) & (De>10.0) & (De<=30.0) & (umu0>0.5591) & (umu0<=0.6427)
    Tddp[b11] = (-37.48*np.power(umu0[b11],2.0) + 44.388*umu0[b11] -12.141)*ade[b11]
    b12 = (umu0<0.999) & (De>10.0) & (De<=30.0) & (umu0>0.6427) & (umu0<=0.7431)
    Tddp[b12] = (0.3501*np.power(umu0[b12],2.0) - 0.1634*umu0[b12] +0.9385)*ade[b12]
    b13 = (umu0<0.999) & (De>10.0) & (De<=30.0) & (umu0>0.7431) & (umu0<=0.8290)
    Tddp[b13] = (-10.237*np.power(umu0[b13],2.0) + 16.165*umu0[b13] -5.3782)*ade[b13]
    b14 = (umu0<0.999) & (De>10.0) & (De<=30.0) & (umu0>0.8290) & (umu0<=0.8829)
    Tddp[b14] = (-145.49*np.power(umu0[b14],2.0) + 247.15*umu0[b14] -104.0)*ade[b14]
    b15 = (umu0<0.999) & (De>10.0) & (De<=30.0) & (umu0>0.8829) & (umu0<=0.9271)
    Tddp[b15] = (-174.18*np.power(umu0[b15],2.0) + 311.61*umu0[b15] -138.42)*ade[b15]
    b16 = (umu0<0.999) & (De>10.0) & (De<=30.0) & (umu0>0.9271) & (umu0<=0.9612)
    Tddp[b16] = (-147.45*np.power(umu0[b16],2.0) + 275.26*umu0[b16] -127.36)*ade[b16]
    b17 = (umu0<0.999) & (De>10.0) & (De<=30.0) & (umu0>0.9612) & (umu0<=0.9902)
    Tddp[b17] = (25.643*np.power(umu0[b17],2.0) - 53.74*umu0[b17] +29.027)*ade[b17]
    b18 = (umu0<0.999) & (De>10.0) & (De<=30.0) & (umu0>0.9902)
    Tddp[b18] = (1294.4*np.power(umu0[b18],2.0) - 2611.5*umu0[b18] +1318.0)*ade[b18]

    bde = 0.0000166*np.power(De,2.0) + 0.00411*De + 0.352
    b19 = (umu0<0.999) & (De>30.0) & (umu0<=0.2079)
    Tddp[b19] = (-8.4358*np.power(umu0[b19],2.0) + 0.6676*umu0[b19] +1.0565)*bde[b19]
    b20 = (umu0<0.999) & (De>30.0) & (umu0>0.2079) & (umu0<=0.3420)
    Tddp[b20] = (-48.564*np.power(umu0[b20],2.0) + 24.929*umu0[b20] -2.1551)*bde[b20]
    b21 = (umu0<0.999) & (De>30.0) & (umu0>0.3420) & (umu0<=0.4383)
    Tddp[b21] = (56.902*np.power(umu0[b21],2.0) - 44.46*umu0[b21] +9.59)*bde[b21]
    b22 = (umu0<0.999) & (De>30.0) & (umu0>0.4383) & (umu0<=0.5591)
    Tddp[b22] = (59.618*np.power(umu0[b22],2.0) - 56.882*umu0[b22] +14.45)*bde[b22]
    b23 = (umu0<0.999) & (De>30.0) & (umu0>0.5591) & (umu0<=0.6427)
    Tddp[b23] = (-225.52*np.power(umu0[b23],2.0) + 265.66*umu0[b23] +77.136)*bde[b23]
    b24 = (umu0<0.999) & (De>30.0) & (umu0>0.6427) & (umu0<=0.7431)
    Tddp[b24] = (8.6548*np.power(umu0[b24],2.0) + 11.085*umu0[b24] +4.5026)*bde[b24]
    b25 = (umu0<0.999) & (De>30.0) & (umu0>0.7431) & (umu0<=0.8290)
    Tddp[b25] = (-50.917*np.power(umu0[b25],2.0) + 80.953*umu0[b25] -31.094)*bde[b25] 
    b26 = (umu0<0.999) & (De>30.0) & (umu0>0.8290) & (umu0<=0.8829)
    Tddp[b26] = (-434.92*np.power(umu0[b26],2.0) + 739.75*umu0[b26] -313.63)*bde[b26] 
    b27 = (umu0<0.999) & (De>30.0) & (umu0>0.8829) & (umu0<=0.9271)
    Tddp[b27] = (-392.67*np.power(umu0[b27],2.0) + 702.16*umu0[b27] -312.95)*bde[b27]
    b28 = (umu0<0.999) & (De>30.0) & (umu0>0.9271) & (umu0<=0.9612)
    Tddp[b28] = (-281.4*np.power(umu0[b28],2.0) + 528.53*umu0[b28] -247.08)*bde[b28]
    b29 = (umu0<0.999) & (De>30.0) & (umu0>0.9612) & (umu0<=0.9902)
    Tddp[b29] = (467.5*np.power(umu0[b29],2.0) - 915.28*umu0[b29] +449.03)*bde[b29]
    b30 = (umu0<0.999) & (De>30.0) & (umu0>0.9902)
    Tddp[b30] = (3872.1*np.power(umu0[b30],2.0) - 7746.4*umu0[b30] +3875.3)*bde[b30]


###Tddcld
    a = 1.8099*np.power(umu0,-0.824)
    b = 7.1094*np.power(umu0,1.7436)

    Tddcld = np.zeros_like(Z)
    a1 = tau<=0.9*taup
    Tddcld[a1] = Tddp[a1]*np.tanh(a[a1]*tau[a1])
    a2 = (tau>0.9*taup) & (tau<taup)
    Tddcld[a2] = Tddp[a2]*np.tanh(0.9*a[a2]*taup[a2]) + Tddp[a2]*(np.tanh(b[a2]/np.power(taup[a2],2.0)) \
                  - np.tanh(0.9*a[a2]*taup[a2]))*(tau[a2]-0.9*taup[a2])/(0.1*taup[a2]) 
    a3 = tau>=taup
    Tddcld[a3] = Tddp[a3]*np.tanh(b[a3]/np.power(tau[a3],2.0))

    return Tddcld


#aa = Pwater(np.array([30.0]), np.array([5.0]), np.array([10.0]) )
#aa = Pwater(np.arange(10)*8.0, np.arange(10), (np.arange(10)+1)*10 )
aa = Pice(np.arange(10)*8.0, np.arange(10), (np.arange(10)+1)*10 )
print(aa)


def farms_dni(F0, tau, solar_zenith_angle, De, phase1, phase2, Tddclr, Ftotal,  Tddcld0, Tddcld1, Fd2 ):

    
    Fd0 = solar_zenith_angle*F0*Tddcld*Tddclr


    return Fd, dni





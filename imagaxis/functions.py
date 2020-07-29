from numpy import *
import numpy as np
import sys
import os

'''
Definitions:

--------
Phil 
--------

D0 = -2*omega/(ivn^2 + omega^2)

D = [D0^(-1) - PI]^(-1)

G0 = [iwn - ek - Sigma]
  
G = [G0^(-1) - Sigma]^(-1)

Hint ~ g * (b^\dagger + b)

Sigma ~ -g^2 * D * G

PI ~ 2*g^2 * G * G

lambda = 2*g^2/(W*omega) = alpha^2 / (K*W)

---------
Marsiglio
---------

Hint ~ alpha / sqrt(2 omega) * (b + b^\dagger)

D = [-(omega^2 + vn^2) - PI]^(-1)

Sigma ~ -alpha^2 D G

PI ~ 2 alpha^2 G G

lambda_dimen = alpha^2/(omega^2)

-------
Beth
-------

Hint ~ g n X = g n (b + b^\dagger) / sqrt(2 omega)

lamb = g^2/(omega^2 * W)

----------------------------

Latest but contradicting result:
    
Hint_beth = g_beth n X = g _beth (b + b^dag) / sqrt(2 omega)
Hint_phil = g_phil n (b + b^dag)

g_phil = g_beth / sqrt(2 omega)

lamb = 2 g_phil^2 / (omega W) = g_beth^2 / (omega^2 W)

----------------------------

g_phil = g_beth * sqrt(2 omega) ?????
lambda_phil = lambda_beth 

alpha_mars = g_beth
plambda_mars = lambda_beth * W = lambda_phil * W

g_phil = g_beth * sqrt(2 omega) ?????

g_phil = alpha_mars * sqrt(2 omega)

The code uses phils definitions for the propagators and selfenergies

So we need to determine g (g_phil) as input for the code...

use lambda = 2*g_phil^2/(omega*W) as definition

so g_phil = sqrt(lambda * omega * W / 2)

----------------------------------------------

relation to marsiglio:
H_m ~ alpha_m / sqrt(2 omega) * (b + b^\dagger)
-> g_phil = alpha_m / sqrt(2 omega)
lamb_m = alpha_m^2 / omega^2 = 2 g_phil^2 / omega
lamb_m = W * lamb_phil

----------------------------------------------

Dm = marsiglio D
Dp = phil D

alpha^2 Dm = gp^2 Dp

Dm = 2*omega Dp

PIm = 2*omega PIp

--------
Ilya
--------

rhoEf = 0.3?
lambda_ilya = lamb_phil * W * rhoEF = lamb_phil * 2.4


'''

#class params:
#    pass

#@classmethod
#    def init(cls):
#        cls.g0 = sqrt(params.lamb * params.omega * params.W / 2.0)


# ParametersI below are all fundamental parameters
# These parameters plus band-structure completely specifies the simulation
# other "parameters" which derive from these are computed on setup

def lamb2g0_ilya(lamb, omega, W):
    # ilya's definition of lamb
    return sqrt(0.5 * lamb / 2.4 * omega * W)

def g02lamb_ilya(g0, omega, W):
    # ilya's definition of lamb
    return g0**2 * 2.4 / (omega * W * 0.5)

def mylamb2g0(lamb, omega, W):
    return sqrt(lamb * omega * W / 2.0)

def myg02lamb(g0, omega, W):
    return 2.0 * g0**2 / (omega * W)

def band(nk, t, tp, Q=None):
    #return -2.0*(cos(kxs) + cos(kys))  #+ alpha**2
    kys, kxs = meshgrid(arange(-pi, pi, 2*pi/nk), arange(-pi, pi, 2*pi/nk))
    ek = -2.0*t*(cos(kxs)+cos(kys)) - 4.0*(tp)*cos(kxs)*cos(kys)
    return ek

def band_1d_lattice(nk, t, tp, Q=None):
    assert tp==0.0
    #return -2.0*(cos(kxs) + cos(kys))  #+ alpha**2
    kxs = arange(-pi, pi, 2*pi/nk)
    ek = -2.0*t*cos(kxs)
 
    if Q is not None:
       ekpq = -2.0*t*cos(kxs+Q)
       ek = (ek, ekpq)

    return ek

def band_square_lattice(nk, t, tp, Q=None):
    #return -2.0*(cos(kxs) + cos(kys))  #+ alpha**2
    kys, kxs = meshgrid(arange(-pi, pi, 2*pi/nk), arange(-pi, pi, 2*pi/nk))
    ek = -2.0*t*(cos(kxs)+cos(kys)) - 4.0*(tp)*cos(kxs)*cos(kys)

    if Q is not None:
        Qx, Qy = Q
        ekpq = -2.0*t*(cos(kxs+Qx)+cos(kys+Qy)) - 4.0*(tp)*cos(kxs+Qx)*cos(kys+Qy)
        ek = (ek, ekpq)

    return ek


def band_1site(*args):
    return 0.1


def gexp_1d(nk, q0):
    qs = arange(-pi, pi, 2*pi/nk)
    return exp(-abs(qs/q0))


def gexp_2d(nk, q0):
    qs = arange(-pi, pi, 2*pi/nk)
    return exp(-sqrt(qs[:,None]**2 + qs[None,:]**2)/q0)

def omega_q(omega, J, nk):
    qs = arange(-pi, pi, 2*pi/nk)
    return omega + J/2*(sin(qs[:,None]/2)**2 + sin(qs[None,:]/2)**2)


'''
params = {}
params['nw']    = 512
params['nk']    = 12
params['t']     = 1.0
params['tp']    = -0.3 # -0.3
params['omega'] = 0.17 # 0.17
params['dens']  = 0.8
params['renormalized'] = True
params['sc']    = 1
params['band']  = band
params['beta']  = 16.0
params['g0']    = 0.125
'''


def read_params(basedir, folder):
        
    params = {}
    for fname in os.listdir(os.path.join(basedir, 'data/', folder)):
        if '.npy' in fname:
            data = np.load(os.path.join(basedir, 'data/', folder, fname), allow_pickle=True)
            params[fname[:-4]] = data

    floats = ['beta', 'dens', 'dw', 'g0', 'mu', 'omega', \
              'idelta', 't', 'tp', 'wmax', 'wmin']
    ints   = ['dim', 'nk', 'nw']
    bools  = ['sc']
    
    for f in floats:
        if f in params:
            params[f] = params[f][0]
    for i in ints:
        if i in ints:
            params[i] = int(params[i][0])
    for b in bools:
        if b in bools:
            params[b] = bool(params[b][0])
        
    params['band'] = None

    return params

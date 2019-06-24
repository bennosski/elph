from numpy import *
import sys

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

D = [-(omega^2 + vn)^2 - PI]^(-1)

Sigma ~ -alpha^2 D G

PI ~ 2 alpha^2 G G

lambda_dimen = alpha^2/(omega^2)

-------
Beth
-------

Hint ~ g n X = g (b + b^\dagger) / sqrt(2 omega)

lamb = g^2/(omega^2 * W)

----------------------------

g_phil = g_beth * sqrt(2 omega)
lambda_phil = lambda_beth 

alpha_mars = g_beth
lambda_mars = lambda_beth * W = lambda_phil * W

g_phil = g_beth * sqrt(2 omega)

g_phil = alpha_mars * sqrt(2 omega)

The code uses phils definitions for the propagators and selfenergies

So we need to determine g (g_phil) as input for the code...

use lambda = 2*g_phil^2/(omega*W) as definition

so g_phil = sqrt(lambda * omega * W / 2)

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

def lamb2g0(lamb, omega, W):
    return sqrt(0.5 * lamb / 2.4 * omega * W)

def g02lamb(g0, omega, W):
    return g0**2 * 2.4 / (omega * W * 0.5)


def band(nk, t, tp):
    #return -2.0*(cos(kxs) + cos(kys))  #+ alpha**2
    kys, kxs = meshgrid(arange(-pi, pi, 2*pi/nk), arange(-pi, pi, 2*pi/nk))

    ek = -2.0*t*(cos(kxs)+cos(kys)) - 4.0*(tp)*cos(kxs)*cos(kys)
    return ek

params = {}
params['nw']    = 1024
params['nk']    = 64
params['t']     = 1.0
params['tp']    = -0.3 # -0.3
params['omega'] = 1.0 # 0.17
params['dens']  = 0.8
params['renormalized'] = True
params['sc']    = 0
params['band']  = band
#params['beta']  = 16.0
#params['g0']    = 0.125

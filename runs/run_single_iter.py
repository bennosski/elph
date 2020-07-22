# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:57:09 2020

@author: 18595
"""
import sys
import os
import src
from migdal_2d import Migdal
from real_2d import RealAxisMigdal
from functions import band_square_lattice, mylamb2g0
import numpy as np
from interpolator import Interp
import matplotlib.pyplot as plt

basedir = '/scratch/users/bln/elph/data/single_iter/'
#basedir = '/scratch/users/bln/elph/data/test/'
if not os.path.exists(basedir): os.makedirs(basedir)
#assert os.path.exists(basedir)



renorm = int(sys.argv[1])

params = {}
params['nw']    =  128
params['nk']    =  120
params['t']     =  1.0
params['tp']    = -0.3                                                                                                 
params['dens']  =  0.8     
#params['fixed_mu'] = -1.11                                                                                            
params['omega'] =  0.17
params['renormalized'] = True if renorm==1 else False
params['sc']    = False
params['band']  = band_square_lattice
params['beta']  = 16.0
params['dim']   = 2
params['g0'] = mylamb2g0(lamb=1/6, omega=params['omega'], W=8.0)
params['Q']  = None

'''
params['dw']     = 0.001
params['wmin']   = -4.2
params['wmax']   = +4.2
params['idelta'] = 0.002j
migdal = Migdal(params, basedir)
'''

params['dw']     = 0.01
params['wmin']   = -7.2
params['wmax']   = +7.2
params['idelta'] = 0.05j





def imag_axis():
    interp = None
    if False:
        interp_folder = '/scratch/users/bln/elph/data/2d/data/data_{}renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'.format('' if renorm==1 else 'un')
        interp = Interp(interp_folder, params['nk'])

    migdal = Migdal(params, basedir)
    migdal.selfconsistency(sc_iter=1, frac=0.9, cont=True, interp=interp)


def real_axis():
    migdal = RealAxisMigdal(params, basedir)
    migdal.selfconsistency(sc_iter=1, frac=1, cont=False)


import single_iter



# RUN THE program
# ----------------
# real_axis()


folder = '/scratch/users/bln/elph/data/single_iter/data/data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone/'

mu = np.load(folder + 'mu.npy')[0]


w = np.arange(params['wmin'], params['wmax'], params['dw'])

width = 0.05j
#S = single_iter.get_S(w, 1/6, params['beta'], params['omega'], params['nk'], params['idelta'], mu)
S = single_iter.get_S(w, 1/6, params['beta'], params['omega'], 120, width, mu)


path = os.path.join(folder, 'SR.npy')

SR = np.load(path)
print(SR.shape)

plt.figure()
plt.plot(w, S.real)
plt.plot(w, S.imag)

nr = len(w)
print('deriv exact sol', (S[nr//2+2] - S[nr//2-2])/(w[nr//2+2] - w[nr//2-2]))


plt.plot(w, SR[0,0].real)
plt.plot(w, SR[0,0].imag)

plt.savefig(basedir + 'S')


nr = len(w)
print('deriv migdal ', (SR[0,0,nr//2+2] - SR[0,0,nr//2-2])/(w[nr//2+2] - w[nr//2-2]))

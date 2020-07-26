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


basedir = '/scratch/users/bln/elph/data/2dfixedn/'
#basedir = '/scratch/users/bln/elph/data/test/'
#basedir = '../'
if not os.path.exists(basedir): os.makedirs(basedir)
#assert os.path.exists(basedir)


params = {}
params['nw']    =  128
params['nk']    =  150
params['t']     =  1.0
params['tp']    = -0.3                                                                                                 
params['dens']  =  0.8     
#params['fixed_mu'] = -1.11                                                                                            
params['omega'] =  1.0
params['renormalized'] = False
params['sc']    = False
params['band']  = band_square_lattice
params['beta']  = 16.0
params['dim']   = 2
params['g0'] = mylamb2g0(lamb=0.1, omega=params['omega'], W=8.0)
params['Q']  = None

params['dw']     = 0.010
params['wmin']   = -4.2
params['wmax']   = +4.2
params['idelta'] = 0.020j
migdal = Migdal(params, basedir)


def imag_axis():
    interp = None
    if False:
        interp_folder = '/scratch/users/bln/elph/data/2d/data/data_{}renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'.format('' if params['renormalized'] else 'un')
        interp = Interp(interp_folder, params['nk'])

    migdal.selfconsistency(sc_iter=200, frac=0.9, cont=True, interp=interp)


def real_axis():
    migdal = RealAxisMigdal(params, basedir)
    migdal.selfconsistency(sc_iter=20, frac=0.8, cont=False)


def plot():
    folder = '/scratch/users/bln/elph/data/2dfixedn/data/data_{}renormalized_nk150_abstp0.300_dim2_g00.63246_nw128_omega1.000_dens0.800_beta16.0000_QNone'.format('' if params['renormalized'] else 'un')

    w = np.load(folder + '/w.npy')
    SR = np.load(folder + '/SR.npy')[0,0]
    
    print(SR.shape)
    
    
    plt.figure()
    plt.plot(w, -SR.real)
    
    h = len(w)//2
    SR -= SR.real[h]
    wm = np.concatenate((w[:h], w[h+1:]))
    Sm = np.concatenate((SR[:h], SR[h+1:]))
    plt.figure()
    plt.plot(wm, -Sm.real/wm)
    


# run the program
# ----------------
#imag_axis()
#real_axis()

plot()







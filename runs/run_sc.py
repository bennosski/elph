# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:57:09 2020

@author: 18595
"""
import os
import src
from migdal_2d_sc import Migdal
from real_2d_sc import RealAxisMigdal
from functions import band_square_lattice, mylamb2g0
import numpy as np
from interpolator import Interp

#basedir = '../'
basedir = '/scratch/users/bln/elph/data/sc2dfixedn'
if not os.path.exists(basedir): os.mkdir(basedir)

params = {}
params['nw']    =  256
params['nk']    =  240
params['t']     =  1.0
params['tp']    = -0.3                                                                                                 
params['dens']  =  0.8                                             #params['fixed_mu'] = -1.11
                                                    
params['omega'] =  0.17
params['renormalized'] = True
params['sc']    = True
params['band']  = band_square_lattice
params['beta']  = 100.0
params['dim']   = 2
params['g0'] = mylamb2g0(lamb=1/6, omega=params['omega'], W=8.0)
params['Q']  = None
params['q0'] = None

params['dw']     = 0.005
params['wmin']   = -4.2
params['wmax']   = +4.2
params['idelta'] = 0.010j


def imag_axis():
    interp = True
    if interp:
        interp_folder = '/scratch/users/bln/elph/data/sc2dfixed/data/data_renormalized_nk240_abstp0.300_dim2_g00.33665_nw256_omega0.170_dens0.800_beta100.0000_QNone'
        interp = Interp(interp_folder, params['nk'])

    migdal = Migdal(params, basedir)
    migdal.selfconsistency(sc_iter=400, frac=0.4, cont=True, interp=interp)


def real_axis():    
    migdal = RealAxisMigdal(params, basedir)
    migdal.selfconsistency(sc_iter=30, fracR=0.4, cont=True)


# run the program
# -----------------
imag_axis()
#real_axis()



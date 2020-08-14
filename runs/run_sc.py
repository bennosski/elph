# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:57:09 2020

@author: 18595
"""
import os
import src
from migdal_2d_sc import Migdal
from real_2d_sc import RealAxisMigdal
from functions import band_square_lattice, mylamb2g0, find_folder
import numpy as np
from interpolator import Interp
import sys

#basedir = '../'
basedir = '/scratch/users/bln/elph/data/sc2dfixed/'
if not os.path.exists(basedir): os.mkdir(basedir)

nk_interp = int(sys.argv[1])
nk = int(sys.argv[2])
beta = float(sys.argv[3])

params = {}
params['nw']    =  256
params['nk']    =  nk
params['t']     =  1.0
params['tp']    = -0.3                                                                                                 

params['dens']  =  0.8
#params['fixed_mu'] = -1.083
params['fixed_mu'] = -1.11
                                                    
params['omega'] =  0.17
params['renormalized'] = True
params['sc']    = True
params['band']  = band_square_lattice
params['beta']  = beta
params['dim']   = 2
params['g0'] = mylamb2g0(lamb=1/6, omega=params['omega'], W=8.0)
params['Q']  = None
params['q0'] = None

'''
params['dw']     = 0.025
params['wmin']   =  -6.0
params['wmax']   = +10.0
params['idelta'] = 0.050j
'''

params['dw']     = 0.010
params['wmin']   =  -4.2
params['wmax']   =   4.2
params['idelta'] = 0.020j


def imag_axis():
    interp = None
    if nk_interp > 0:
        interp_folder = '/scratch/users/bln/elph/data/sc2dfixed/data/data_renormalized_nk{}_abstp0.300_dim2_g00.33665_nw256_omega0.170_dens0.800_beta100.0000_QNone'.format(nk_interp)
        interp = Interp(interp_folder, params['nk'])

    migdal = Migdal(params, basedir)
    migdal.selfconsistency(sc_iter=60, frac=0.4, cont=True, interp=interp)


def real_axis():    
    
    interp = None
    if True:
        #interp_folder = '/scratch/users/bln/elph/data/sc2dfixed/data/data_renormalized_nk{}_abstp0.300_dim2_g00.33665_nw256_omega0.170_dens0.800_beta100.0000_QNone'.format(nk_interp)

        ps = {'renormalized': True, 'nk': 200, 'omega': 0.17, 'beta': beta, 'idelta': 0.050, 'wmin': -4.2, 'wmax': 4.2}

        interp_folder, _ = find_folder(basedir, ps)

        print('interp folder', interp_folder)

        w = np.arange(params['wmin'], params['wmax'] + params['dw']/2, params['dw'])
        interp = Interp(interp_folder, w, kind='frequency')
        

    migdal = RealAxisMigdal(params, basedir)
    migdal.selfconsistency(sc_iter=40, fracR=0.4, cont=True, interp=interp)


# run the program
# -----------------
#imag_axis()
real_axis()




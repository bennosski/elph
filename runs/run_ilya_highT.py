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
from functions import band_square_lattice, mylamb2g0, find_folder
import numpy as np
from interpolator import Interp

#basedir = '/scratch/users/bln/elph/data/2dfixedn/'
#basedir = '/scratch/users/bln/elph/data/test/'
basedir = '/scratch/users/bln/elph/data/2dn0p786/'
if not os.path.exists(basedir): os.makedirs(basedir)
#assert os.path.exists(basedir)

renorm = int(sys.argv[1])

params = {}
params['nw']    =  128
params['nk']    =  120
params['t']     =  1.0
params['tp']    = -0.3                                                                                                 
params['dens']  =  0.786     
#params['fixed_mu'] = -1.11                                                                                            
params['omega'] =  0.17
params['renormalized'] = True if renorm==1 else False
params['sc']    = False
params['band']  = band_square_lattice
params['beta']  = 16.0
params['dim']   = 2
params['g0'] = mylamb2g0(lamb=1/6, omega=params['omega'], W=8.0)
params['Q']  = None

params['dw']     = 0.005
params['idelta'] = 0.010j
params['wmin'] = -5.0
params['wmax'] =  9.0

#params['wmin'] = -1.9
#params['wmax'] =  6.5
#params['wmin']   = -4.2
#params['wmax']   = +4.2
#params['idelta'] = 0.002j
migdal = Migdal(params, basedir)


def imag_axis():
    interp = None
    if True:
        interp_folder = '/scratch/users/bln/elph/data/2d/data/data_{}renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'.format('' if renorm==1 else 'un')
        interp = Interp(interp_folder, params['nk'])

    migdal.selfconsistency(sc_iter=200, frac=0.9, cont=True, interp=interp)


def real_axis():
    interp = None
    if True:
        ps = {'renormalized': True, 'nk': 120, 'nw': 128, 'omega': 0.170, 'beta': 16.0, 'idelta': 0.020}
        interp_folder = find_folder(basedir, ps)
        #interp_folder = '/scratch/users/bln/elph/data/2dn0p786/data/data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.786_beta16.0000_QNone_idelta0.0200'

        return

        w = np.arange(params['wmin'], params['wmax'] + params['dw']/2, params['dw'])
        interp = Interp(interp_folder, w, kind='frequency')

    migdal = RealAxisMigdal(params, basedir)
    migdal.selfconsistency(sc_iter=40, frac=0.8, cont=True, interp=interp)


# run the program
# ----------------
#imag_axis()
real_axis()



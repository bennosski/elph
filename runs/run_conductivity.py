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

#basedir = '/scratch/users/bln/elph/data/2dfixedn/'
basedir = '../data/conductivity/'
if not os.path.exists(basedir): os.makedirs(basedir)
#assert os.path.exists(basedir)


def run(renorm, beta, S0, PI0):

    params = {}
    params['nw']    =  128
    params['nk']    =   12
    params['t']     =  1.0
    params['tp']    = -0.3                                                                                                 
    params['dens']  =  0.8     
    #params['fixed_mu'] = -1.11                                                                                            
    params['omega'] =  0.17
    params['renormalized'] = True if renorm==1 else False
    params['sc']    = False
    params['band']  = band_square_lattice
    params['beta']  = beta
    params['dim']   = 2
    params['g0'] = mylamb2g0(lamb=1/6, omega=params['omega'], W=8.0)
    params['Q']  = None

    params['dw']     = 0.001
    params['wmin']   = -4.2
    params['wmax']   = +4.2
    params['idelta'] = 0.002j
    migdal = Migdal(params, basedir)


    savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter=200, frac=0.4, cont=True, S0=S0, PI0=PI0)
    migdal.compute_jjcorr(savedir, G, D)
    PI = params['g0']**2 * GG

    migdal = RealAxisMigdal(params, basedir)
    migdal.selfconsistency(sc_iter=35, frac=0.8, cont=False)

    path = os.path.join(savedir, 'G.npy')
    G  = np.load(path)
    path = os.path.join(savedir, 'GR.npy')
    GR = np.load(path)
    migdal.compute_jjcorr(savedir, G, GR)

    return S, PI


# run the program
# ----------------


for renorm in (1,0):
    beta = 1
    dbeta = 1
    S0, PI0 = None, None

    while beta <= 16:
        S0, PI0 = run(renorm, beta, S0, PI0)

        if S0 is None:
            print('failed to converge')
            exit()

        beta += dbeta

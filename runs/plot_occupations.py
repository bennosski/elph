# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:32:10 2020

@author: ben
"""

#
# plot the phonon occupation
#

import src
import os
from functions import read_params
import numpy as np
import matplotlib.pyplot as plt


def el_occ(basedir, folder):
    # electronic occupation as a test

    params = read_params(basedir, folder)
    
    #print(params['wmin'], params['wmax'])
    w = np.arange(params['wmin'], params['wmax'], params['dw'])
    
    G = np.load(os.path.join(basedir, 'data', folder, 'GR.npy'))
    #G = np.load(basedir + folder + '/GR.npy')
    
    nF = 1/(np.exp(params['beta']*w) + 1)
    norm = -1/np.pi * np.trapz(G.imag, dx=params['dw'], axis=2)
    nk = -1/np.pi * np.trapz(nF[None,None,:]*G.imag, dx=params['dw'], axis=2)

    plt.figure()
    plt.imshow(nk, origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig(basedir + 'nk')
    plt.close()
    
    plt.figure()
    plt.imshow(norm, origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig(basedir + 'norm')
    plt.close()
    

def ph_occ(basedir, folder):
    # electronic occupation as a test

    params = read_params(basedir, folder)
    
    #print(params['wmin'], params['wmax'])
    w = np.arange(params['wmin'], params['wmax'], params['dw'])
    
    D = np.load(os.path.join(basedir, 'data', folder, 'DR.npy'))
    
    nB = 1/(np.exp(params['beta']*w) - 1)
    N = -1/np.pi * np.trapz(nB[None,None,:]*D.imag, dx=params['dw'], axis=2)
    N = (N - 1)/2
    
    plt.figure()
    plt.imshow(N, origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig(basedir + 'N')
    plt.close()
    




basedir = '../'
#folder = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
folder = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
#el_occ(basedir, folder)
ph_occ(basedir, folder)

    
    
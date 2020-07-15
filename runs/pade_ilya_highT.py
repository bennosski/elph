# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:42:23 2020

@author: ben
"""

import src
import pade
import os
import numpy as np
import fourier
import matplotlib.pyplot as plt

folder = '../data/data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    
D = np.load(folder + '/D.npy')
print(D.shape)
beta = np.load(folder + '/beta.npy')[0]
nk = D.shape[0]

def get_pade(D, beta, ikx, iky, w):
        
    Dt = D[ikx, iky]
    
    Dw = fourier.t2w(Dt, beta, 0, 'boson')
    nw = len(Dw)
    ivns = 1j*2*np.arange(nw)*np.pi/beta
    
    #print('shape Dw', Dw.shape)
    #print('ivns shape', ivns.shape)
    
    # extend to negative freqs
    # (would be different for fermions)
    ivns = np.concatenate((-ivns[1:][::-1], ivns)) 
    Dw = np.concatenate((Dw[1:][::-1], Dw))
    
    #print('ivns shape', ivns.shape)
    #print('Dw shape', Dw.shape)
    
    #plt.figure()
    #plt.plot(ivns.real, Dw.real)
    #plt.title('Dw')
    
    i0 = nw - 0
    i1 = nw + 10
    
    inds = range(i0, i1, 1)
    
    #print('zs ', ivns[inds])
    
    p = pade.fit(ivns[inds], Dw[inds])
    
    return -1.0/np.pi * np.array([p(x) for x in w]).imag

 
def compute_B():    
    w = np.linspace(0, 0.2, 500)
    
    B = []
    for i in range(nk//2):
        B.append(get_pade(D, beta, nk//2-i, nk//2, w))
    for i in range(nk//2):
        B.append(get_pade(D, beta, 0, nk//2-i, w))
    for i in range(nk//2):
        B.append(get_pade(D, beta, i, i, w))
    B.append(B[0])
    B = np.array(B)
    
    np.save(folder + 'Bpade_0_10_1', B)


def plot_B():
    B = np.load(folder + 'Bpade.npy')
    #B = np.load(folder + 'Bpade_0_10_1.npy')
    
    plt.figure()
    plt.imshow(B.T, origin='lower', aspect='auto', cmap='Greys', vmin=0,
               interpolation='bilinear', extent=[-np.pi,np.pi,0,0.2])
    plt.ylim(0, 0.17)
    #plt.colorbar()
    plt.savefig(folder + '/Bpade.png')


#compute_B()
plot_B()

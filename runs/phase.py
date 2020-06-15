# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 10:23:53 2020

@author: 18595
"""

import os
import src
from migdal_2d import Migdal
from real_2d import RealAxisMigdal
from functions import band_square_lattice, mylamb2g0
import numpy as np
from interpolator import Interp
import matplotlib.pyplot as plt

basedir = '../test_phase/'
if not os.path.exists(basedir): os.mkdir(basedir)

params = {}
params['nw']    =  128
params['nk']    =   16
params['t']     =  1.0
params['tp']    = -0.3                                                                                                
params['dens']  =  0.8                                                                                                 
params['omega'] =  0.17
params['renormalized'] = False
params['sc']    = False
params['band']  = band_square_lattice
params['beta']  = 1.0
params['dim']   = 2
params['g0'] = mylamb2g0(lamb=1/6, omega=params['omega'], W=8.0)
#params['Q']  = None
#params['q0'] = None
params['fixed_mu'] = -1.11

params['dw']     = 0.005
params['wmin']   = -4.2
params['wmax']   = +4.2
params['idelta'] = 0.025j


def read_params(basedir, folder):
    
    params = {}
    for fname in os.listdir(os.path.join(basedir, 'data', folder)):
        if '.npy' in fname:
            data = np.load(os.path.join(basedir, 'data', folder, fname), allow_pickle=True)
            params[fname[:-4]] = data

    floats = ['beta', 'dens', 'dw', 'g0', 'mu', 'omega', \
              'idelta', 't', 'tp', 'wmax', 'wmin']
    ints   = ['dim', 'nk', 'nw']
    bools  = ['sc']
    
    for f in floats:
        params[f] = params[f][0]
    for i in ints:
        params[i] = int(params[i][0])
    for b in bools:
        params[b] = bool(params[b][0])
        
    params['band'] = band_square_lattice

    return params


def test1():
    migdal = Migdal(params, basedir)
    savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter=200, frac=0.6, cont=True)
    Xsc, Xcdw = migdal.susceptibilities(500, G, D, GG, frac=0.7)
    print('Xsc', Xsc)
        

def x_vs_t():
    
    params['beta'] = 1.0
    dbeta = 2.0
    #params['beta'] = 85.0
    #dbeta = 1.0
    S0, PI0 = None, None
    
    while dbeta > 0.2 and params['beta']<15:
        
        migdal = Migdal(params, basedir)
        savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter=400, frac=0.2, S0=S0, PI0=PI0, cont=True)
        
        if G is not None:
            params['beta'] += dbeta
            S0 = S
            PI0 = GG * params['g0']**2
            
            Xsc, Xcdw = migdal.susceptibilities(1000, G, D, GG, frac=1)
            print('Xsc', Xsc)
            np.save(savedir + 'Xsc.npy', [Xsc])
            np.save(savedir + 'Xcdw.npy', Xcdw)
        else:
            dbeta /= 2
            params['beta'] -= dbeta
        
     
def plot_x_vs_t():
    df = os.path.join(basedir, 'data/')
    folders = os.listdir(df)
    
    data = {}
    
    data['renormalized']   = {'betas':[], 'xscs': [], 'xcdws': []}
    data['unrenormalized'] = {'betas':[], 'xscs': [], 'xcdws': []}
    
    for folder in folders:
        params = read_params(basedir, folder)
        path = os.path.join(df, folder, 'Xsc.npy')
        pathcdw = os.path.join(df, folder, 'Xcdw.npy')
        
        if not os.path.exists(path): continue
        
        
        xsc = np.load(path, allow_pickle=True)[0]
        xcdw = np.load(pathcdw, allow_pickle=True)
        print('argmax', np.argmax(xcdw), 'shape xcdw', np.shape(xcdw))
        xcdw = np.amax(xcdw)
        r = 'renormalized' if params['renormalized'] else 'unrenormalized'
        
        if xsc is not None:
            data[r]['betas'].append(params['beta'])
            data[r]['xscs'].append(xsc)
            data[r]['xcdws'].append(xcdw)

    d = data['renormalized']
    rb, rx, rc = zip(*sorted(zip(d['betas'], d['xscs'], d['xcdws'])))
    d = data['unrenormalized']
    ub, ux, uc = zip(*sorted(zip(d['betas'], d['xscs'], d['xcdws'])))
    
    rb = np.array(rb)
    rx = np.array(rx)
    rc = np.array(rc)
    
    ub = np.array(ub)
    ux = np.array(ux)
    uc = np.array(uc)
    
    
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.set_size_inches(6, 3)
    plt.title('lamb=1/6, n=0.8, 16x16, t\'=-0.3')
    #plt.title('$\lambda_0=2$'+', $\Omega=1$,'+r'$\langle n \rangle$'+'=0.8')
   
    ax1.plot(1/ub, 1/ux, 'k-')
    ind = uc!=None
    ax1.plot(1/ub[ind], 20/uc[ind], 'k--')
    ax1.legend(['1/Xsc', '20/Xcdw'])
    ax1.set_title('unrenormalized ME')
    ax1.set_xlabel('T', fontsize=13)
     
    ax1.set_xlim(0, 0.6)
    ax1.set_ylim(0, 6)
    
    ax2.plot(1/rb, 1/rx, 'k-')
    ind = rc!=None
    ax2.plot(1/rb[ind], 20/rc[ind], 'k--')
    
    ax2.set_xlim(0, 0.05)
    ax2.set_ylim(0, 1.5)
    ax2.legend(['1/Xsc', '20/Xcdw'])
    ax2.set_title('renormalized ME')
    ax2.set_xlabel('T', fontsize=13)
    
    
    plt.tight_layout()
 
    
    plt.savefig(basedir+'div')
    plt.close()
        
    print(uc)
    print('beta and x r', [(b,x) for b,x in zip(rb, rc)])
 
#test1()
#x_vs_t()
plot_x_vs_t()



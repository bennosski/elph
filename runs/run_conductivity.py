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

#basedir = '/scratch/users/bln/elph/data/2dfixedn/'
basedir = '../data/conductivity/'
if not os.path.exists(basedir): os.makedirs(basedir)
#assert os.path.exists(basedir)


params = {}
params['nw']    =  128
params['nk']    =   12
params['t']     =  1.0
params['tp']    = -0.3                                                                                                 
params['dens']  =  0.8     
#params['fixed_mu'] = -1.11                                                                                            
params['omega'] =  0.17

params['sc']    = False
params['band']  = band_square_lattice

params['dim']   = 2
params['g0'] = mylamb2g0(lamb=1/6, omega=params['omega'], W=8.0)
params['Q']  = None

params['dw']     = 0.001
params['wmin']   = -4.2
params['wmax']   = +4.2
params['idelta'] = 0.002j



def run(renorm, beta, S0, PI0):
    params['renormalized'] = True if renorm==1 else False
    params['beta']  = beta
    
    migdal = Migdal(params, basedir)

    savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter=200, frac=0.4, cont=True, S0=S0, PI0=PI0)
    print('compute jjcorr ivn')
    migdal.compute_jjcorr(savedir, G, D)
    PI = params['g0']**2 * GG

    migdal = RealAxisMigdal(params, basedir)
    migdal.selfconsistency(sc_iter=35, frac=0.8, cont=False)

    print('compute jjcorr w')
    path = os.path.join(savedir, 'G.npy')
    G  = np.load(path)
    path = os.path.join(savedir, 'GR.npy')
    GR = np.load(path)
    migdal.compute_jjcorr(savedir, G, GR)

    return S, PI


def full_run():
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
    
    
def run_betas_dqmc():
    betas = [0.8, 1.6, 2.8, 4.8]
    
    for renorm in (1,0):
        S0, PI0 = None, None
    
        for beta in betas:
            S0, PI0 = run(renorm, beta, S0, PI0)
    
            if S0 is None:
                print('failed to converge')
                exit()




class plotter:

    def __init__(self, params):
        self.basedir = basedir
        self.keys = params.keys() 
        for key in params:
            setattr(self, key, params[key])


    def get_savedir(self):
        return self.basedir+'data/data_{}_nk{}_abstp{:.3f}_dim{}_g0{:.5f}_nw{}_omega{:.3f}_dens{:.3f}_beta{:.4f}_Q{}/'.format('renormalized' if self.renormalized else 'unrenormalized', self.nk, abs(self.tp), self.dim, self.g0, self.nw, self.omega, self.dens, self.beta, None)
        
   
    def plot(self):
        
        data = {}
        betas = np.array(range(1, 17)) 
        for renorm in (1, 0):
            data[renorm] = []
            
            for beta in betas:
                 self.renormalized = True if renorm else False
                 self.beta = beta
                 savedir = self.get_savedir()

                 jjv = np.load(savedir + 'jj0v.npy')
                 data[renorm].append(jjv)
    
        plt.figure()
        plt.plot(data[0][-1], '.')
    
        u = np.array(data[0])
        r = np.array(data[1])
           
        plt.figure()
        plt.plot(betas, u[:,0].real, '.-')
        plt.plot(betas, r[:,0].real, '.-')
        plt.xlabel('1/T')
    
    
        data = {}
        betas = np.array(range(1, 17)) 
        for renorm in (1, 0):
            data[renorm] = []
            
            for beta in betas:
                 self.renormalized = True if renorm else False
                 self.beta = beta
                 savedir = self.get_savedir()
                 
                 jjw = np.load(savedir + 'jj0w.npy')
                 data[renorm].append(jjw)
  
        print('shape data', np.array(data[1]).shape)
        
        w = np.arange(self.wmin, self.wmax, self.dw)
        h = len(w)//2
         
        betas = np.array(betas)
         
        def get_conds(w, jjws):
            h = len(w)//2
            conds = []
            for jjw in jjws:
                conds.append(-jjw[h+1].imag / w[h+1])
            return np.array(conds)
             
        uconds = get_conds(w, data[0])
        rconds = get_conds(w, data[1])
            
        plt.figure()
        plt.plot(1./betas, 1/uconds, '-')
        plt.plot(1./betas, 1/rconds, '-')
        
        #xs = np.array([16.0, 12.0, 8.0, 4.8, 2.4, 1.6, 0.8])
        #plt.plot(1/xs, [6]*len(xs), '.')
        
        plt.ylabel('resistivity')
        plt.xlabel('T')
        
        print('slope u', (1/uconds[1] - 1/uconds[0])/(1/betas[1] - 1/betas[0]))
        print('slope r', (1/rconds[1] - 1/rconds[0])/(1/betas[1] - 1/betas[0]))
        
    
        '''
        plt.figure()
        for x in data[0]:
            plt.plot(w[h+1:], -x[h+1:].imag / w[h+1:])
            plt.plot(w[:h], -x[:h].imag / w[:h])
        plt.xlim(-2, 2)
        
    
        plt.figure()
        for x in data[1]:
            plt.plot(w[h+1:], -x[h+1:].imag / w[h+1:])
        plt.xlim(0, 2)
        '''
    
        
        '''
        plt.figure()
        arr = np.array(data[0])   
        plt.plot(betas, arr[:,0], '.-')
        arr = np.array(data[1])   
        plt.plot(betas, arr[:,0], '.-')
        '''   
                 
        

# run the program
# ----------------
#full_run()
run_betas_dqmc()


#p = plotter(params)
#p.plot()

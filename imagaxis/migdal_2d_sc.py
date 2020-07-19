from numpy import *
import numpy as np
from convolution import conv
from base import MigdalBase
from functions import mylamb2g0, band_square_lattice
import os
import time
import sys

class Migdal(MigdalBase):
    tau0 = array([[1.0, 0.0], [0.0, 1.0]])
    tau1 = array([[0.0, 1.0], [1.0, 0.0]])
    tau3 = array([[1.0, 0.0], [0.0,-1.0]])

    #------------------------------------------------------------
    def setup(self):
        out = super().setup()
        assert self.dim==2
        return out
    #------------------------------------------------------------
    def compute_fill(self, Gw):
        return 1.0 + 2./(self.nk**2 * self.beta) * 2.0*sum(Gw[...,0,0]).real # second 2 accounting for negative matsubara freqs
    #------------------------------------------------------------
    def compute_n(self, Gtau):
        return -2.0*mean(Gtau[...,-1,0,0]).real
    #------------------------------------------------------------
    def compute_G(self, wn, ek, mu, S):
        return linalg.inv(1j*wn[None,None,:,None,None]*Migdal.tau0[None,None,None,:,:] - (ek[:,:,None,None,None]-mu)*Migdal.tau3[None,None,None,:,:] - S)
    #------------------------------------------------------------
    def compute_D(self, vn, PI):
        return 1.0/(-((vn**2)[None,None,:] + self.omega**2)/(2.0*self.omega) - PI)
    #------------------------------------------------------------
    def compute_S(self, G, D):
        if hasattr(self, 'gk2'):
            tau3Gtau3 = einsum('ab,kqwbc,cd->kqwad', Migdal.tau3, G, Migdal.tau3)
            return -self.g0**2/self.nk**2 * self.gk2[:,:,None,None,None] * conv(tau3Gtau3, D[:,:,:,None,None], ['k-q,q','k-q,q'], [0,1], [True,True], beta=self.beta)
        else:
            tau3Gtau3 = einsum('ab,kqwbc,cd->kqwad', Migdal.tau3, G, Migdal.tau3)
            return -self.g0**2/self.nk**2 * conv(tau3Gtau3, D[:,:,:,None,None], ['k-q,q','k-q,q'], [0,1], [True,True], beta=self.beta)
    #------------------------------------------------------------
    def compute_GG(self, G):
        if hasattr(self, 'gk2'):
            tau3G = einsum('ab,kqwbc->kqwac', Migdal.tau3, G)
            return 1.0/self.nk**2 * trace(conv(self.gk2[:,:,None,None,None]*tau3G, -tau3G[:,:,::-1], ['k,k+q','k,k+q'], [0,1], [True,True], beta=self.beta, op='...ab,...bc->...ac'), axis1=-2, axis2=-1)
        else:
            tau3G = einsum('ab,kqwbc->kqwac', Migdal.tau3, G)
            return 1.0/self.nk**2 * trace(conv(tau3G, -tau3G[:,:,::-1], ['k,k+q','k,k+q'], [0,1], [True,True], beta=self.beta, op='...ab,...bc->...ac'), axis1=-2, axis2=-1)
    #------------------------------------------------------------
    def init_selfenergies(self):
        S  = zeros([self.nk,self.nk,self.ntau,2,2], dtype=complex)
        PI = zeros([self.nk,self.nk,self.ntau], dtype=complex)
        if self.sc:
           S[...,0,0,1] = 1e-2
           S[...,0,1,0] = 1e-2
           
           if hasattr(self, 'gk2'):
               S *= self.gk2[:,:,None,None,None]
           
        return S, PI
    #------------------------------------------------------------
    def compute_x0(self, F0x, D, jumpF0, jumpD):
        raise NotImplementedError
        #return -self.g0**2/self.nk * conv(F0x, D, ['k-q,q', 'm,n-m'], [0,1], [True,False], self.beta, kinds=('fermion','boson','fermion'), op='...,...', jumps=(jumpF0, jumpD))


if __name__=='__main__':
    time0 = time.time()

    def run():
        # example usage as follows :
        print('2D Migdal with superconductivity')
    
        params = {}
        params['nw']    = 512
        params['nk']    = 40
        params['t']     = 1.0
        params['tp']    = -0.3
        params['omega'] = 0.1
        params['dens']  = 0.8
        params['renormalized'] = True
        params['sc']    = True
        params['band']  = band_square_lattice
        #params['beta']  = 1.0
        params['Q'] = None
        #params['g0']    = 0.525
        #params['dim']   = 2
    
        #basedir = '/home/groups/simes/bln/data/elph/imagaxis/example/'
        
        basedir = '/scratch/users/bln/elph/data/sc2d_200223/'
        if not os.path.exists(basedir): os.makedirs(basedir)
    
        #lamb = 0.3
        #lamb = float(sys.argv[1])
        lamb = 0.2
        print('lamb = ', lamb)
        W    = 8.0
        params['g0'] = mylamb2g0(lamb, params['omega'], W)
        print('g0 is ', params['g0'])
    
        params['beta'] = 5.0
    
        path = sys.argv[1]
        if path:
            S0  = load(path+'S.npy')
            PI0 = load(path+'PI.npy')
            mu0 = load(path+'mu.npy')[0]
        else:
            S0, PI0, mu0 = None, None, None
    
        migdal = Migdal(params, basedir)
    
        sc_iter = 2000
        savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, mu0=mu0, frac=0.1)
    
        if not params['renormalized']:
            PI = None
        else:
            PI = params['g0']**2 * GG
        save(savedir + 'S.npy', S)
        save(savedir + 'PI.npy', PI)
        save(savedir + 'G.npy', G)
        save(savedir + 'D.npy', D)
    
        print('------------------------------------------')
        print('simulation took', time.time()-time0, 's')
        
        
    def run_dwave():
        # example usage as follows :
        print('2D Migdal with superconductivity')
    
        basedir = '../test_dwave2/'
        if not os.path.exists(basedir): os.makedirs(basedir)
    
        params = {}
        params['nw']    = 128
        params['nk']    = 16
        params['t']     = 1.0
        params['tp']    = -0.3
        params['omega'] = 0.17
        params['dens']  = 0.8
        params['renormalized'] = True
        params['sc']    = True
        params['band']  = band_square_lattice
        params['beta']  = 40.0
        params['fixed_mu'] = -1.11
        
        lamb = 1/6
        W    = 8.0
        params['g0'] = mylamb2g0(lamb, params['omega'], W)
        print('g0 is ', params['g0'])
    
        ks = np.arange(-np.pi, np.pi, 2*np.pi/params['nk'])
        params['gk2'] = 2*(0.5*(np.cos(ks[:,None]) - np.cos(ks[None,:])))**2
    
        migdal = Migdal(params, basedir)
    
        sc_iter = 2000
        savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=None, PI0=None, frac=0.2)
    
        print('------------------------------------------')
        print('simulation took', time.time()-time0, 's')



    run_dwave()

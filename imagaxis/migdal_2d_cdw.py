from numpy import *
import numpy as np
from convolution import conv
from base import MigdalBase
from functions import mylamb2g0, band_square_lattice
import os
import time

class Migdal(MigdalBase):
    tau0 = array([[1.0, 0.0], [0.0, 1.0]])
    tau1 = array([[0.0, 1.0], [1.0, 0.0]])
    tau3 = array([[1.0, 0.0], [0.0,-1.0]])
    I00  = array([[1.0, 0.0],  [0.0, 0.0]])
    I11  = array([[0.0, 0.0],  [0.0, 1.0]])

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
    def compute_G(self, wn, eks, mu, S):
        ek, ekpq = eks
        ginv = 1j*wn[None,None,:,None,None]*Migdal.tau0[None,None,None,:,:] - (ek - mu)[:,:,None,None,None]*Migdal.I00[None,None,None,:,:] - (ekpq - mu)[:,:,None,None,None]*Migdal.I11[None,None,None,:,:] - S
        return linalg.inv(ginv)
    #------------------------------------------------------------
    def compute_D(self, vn, PI):
        if hasattr(self, 'omega_q'):
            return 1.0/(-((vn**2)[None,None,:,None,None] + self.omega_q[:,:,None,None,None]**2)/(2.0*self.omega_q[:,:,None,None,None]) - PI)
        else:
            return 1.0/(-((vn**2)[None,None,:,None,None] + self.omega**2)/(2.0*self.omega) - PI)
    #------------------------------------------------------------
    def compute_S(self, G, D):
        #return -self.g0**2/self.nk**2 * conv(G, D, ['k-q,q','k-q,q'], [0,1], [True,True], beta=self.beta)
        out = np.zeros((self.nk,self.nk,self.ntau,2,2), dtype=complex)
        idxs = [(0,0,0,0), (1,0,1,0), (0,1,0,1), (1,1,1,1)]
        for i,j,a,b in idxs:
            out[:,:,:,0,0] += conv(G[:,:,:,i,j], D[:,:,:,a,b], ['k-q,q','k-q,q'], [0,1], [True,True], beta=self.beta)
        idxs = [(0,0,1,0), (1,0,0,0), (0,1,1,1), (1,1,0,1)]
        for i,j,a,b in idxs:
            out[:,:,:,0,1] += conv(G[:,:,:,i,j], D[:,:,:,a,b], ['k-q,q','k-q,q'], [0,1], [True,True], beta=self.beta)
        idxs = [(0,0,0,1), (1,0,1,1), (0,1,0,0), (1,1,1,0)]
        for i,j,a,b in idxs:
            out[:,:,:,1,0] += conv(G[:,:,:,i,j], D[:,:,:,a,b], ['k-q,q','k-q,q'], [0,1], [True,True], beta=self.beta)
        idxs = [(0,0,1,1), (1,0,0,1), (0,1,1,0), (1,1,0,0)]
        for i,j,a,b in idxs:
            out[:,:,:,1,1] += conv(G[:,:,:,i,j], D[:,:,:,a,b], ['k-q,q','k-q,q'], [0,1], [True,True], beta=self.beta)
        
        # factors of 2?
        # needs to agree with normal state when things are diagonal
        return - (1/2) * self.g0**2/self.nk**2 * out
    #------------------------------------------------------------
    def compute_GG(self, G):
        out = np.zeros((self.nk,self.nk,self.ntau,2,2), dtype=complex)
        Grev = -G[:,:,::-1]
        idxs = [(0,0,0,0), (1,0,1,0), (0,1,0,1), (1,1,1,1)]
        for i,j,a,b in idxs:
            out[:,:,:,0,0] += conv(G[:,:,:,i,j], Grev[:,:,:,a,b], ['k,k+q','k,k+q'], [0,1], [True,True], beta=self.beta)
        idxs = [(0,0,1,0), (1,0,0,0), (0,1,1,1), (1,1,0,1)]
        for i,j,a,b in idxs:
            out[:,:,:,0,1] += conv(G[:,:,:,i,j], Grev[:,:,:,a,b], ['k,k+q','k,k+q'], [0,1], [True,True], beta=self.beta)
        idxs = [(0,0,0,1), (1,0,1,1), (0,1,0,0), (1,1,1,0)]
        for i,j,a,b in idxs:
            out[:,:,:,1,0] += conv(G[:,:,:,i,j], Grev[:,:,:,a,b], ['k,k+q','k,k+q'], [0,1], [True,True], beta=self.beta)
        idxs = [(0,0,1,1), (1,0,0,1), (0,1,1,0), (1,1,0,0)]
        for i,j,a,b in idxs:
            out[:,:,:,1,1] += conv(G[:,:,:,i,j], Grev[:,:,:,a,b], ['k,k+q','k,k+q'], [0,1], [True,True], beta=self.beta)
        return -self.g0**2/self.nk**2 * out

    #------------------------------------------------------------
    def init_selfenergies(self):
        S  = zeros([self.nk,self.nk,self.ntau,2,2], dtype=complex)
        PI = zeros([self.nk,self.nk,self.ntau,2,2], dtype=complex)
        if hasattr(self, 'cdw') and self.cdw:
           S[...,0,0,1] = +0.0
           S[...,0,1,0] = +0.0
        return S, PI
    #------------------------------------------------------------
    def compute_x0(self, F0x, D, jumpF0, jumpD):
        raise NotImplementedError
        #return -self.g0**2/self.nk * conv(F0x, D, ['k-q,q', 'm,n-m'], [0,1], [True,False], self.beta, kinds=('fermion','boson','fermion'), op='...,...', jumps=(jumpF0, jumpD))


if __name__=='__main__':
    time0 = time.time()

    # example usage as follows :
    print('2D Migdal with CDW')

    params = {}
    params['nw']    = 128
    params['nk']    = 20
    params['t']     = 1.0
    params['tp']    = 0.0
    params['omega'] = 0.0
    params['dens']  = 1.0
    params['renormalized'] = True
    
    params['cdw']   = True
    params['Q']     = (pi, pi)
    params['band']  = band_square_lattice
    params['beta']  = 1.0
    params['dim']   = 2
    
    ks = np.arange(-np.pi, np.pi, 2*np.pi/params['nk'])
    params['omega_q'] = np.abs(np.sin(0.5*(ks[:,None] - np.pi))) + \
                        np.abs(np.sin(0.5*(ks[None,:] - np.pi)))
    

    #basedir = '/home/groups/simes/bln/data/elph/imagaxis/example2/'
    basedir = '../'
    if not os.path.exists(basedir): os.makedirs(basedir)

    #lamb = 0.1
    #W = 8.0
    #params['g0'] = mylamb2g0(lamb, 1, W)
    #print('g0 is ', params['g0'])
    params['g0'] = 0.1
    
    migdal = Migdal(params, basedir)

    sc_iter = 2000
    S0, PI0, mu0  = None, None, None
    savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, mu0=None, frac=0.2)
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


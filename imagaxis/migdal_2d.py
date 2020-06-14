from numpy import *
from convolution import conv
from base import MigdalBase
from functions import lamb2g0_ilya, band_square_lattice, gexp_2d, omega_q
import os
import time
import matplotlib.pyplot as plt
import numpy as np

class Migdal(MigdalBase):
    #------------------------------------------------------------
    def compute_fill(self, Gw):
        return real(1.0 + 2./(self.nk**2 * self.beta) * 2.0*sum(Gw))
    #------------------------------------------------------------
    def compute_n(self, Gtau):
        return -2.0*mean(Gtau[:,:,-1]).real
    #------------------------------------------------------------
    def compute_G(self, wn, ek, mu, S):
        return 1.0/(1j*wn[None,None,:] - (ek[:,:,None]-mu) - S)
    #------------------------------------------------------------
    def compute_D(self, vn, PI):
        if hasattr(self, 'omega_q'):
            return 1.0/(-((vn**2)[None,None,:] + self.omega_q[:,:,None]**2)/(2.0*self.omega_q[:,:,None]) - PI)
        else:
            return 1.0/(-((vn**2)[None,None,:] + self.omega**2)/(2.0*self.omega) - PI)
    #------------------------------------------------------------
    def compute_S(self, G, D):
        if hasattr(self, 'gq2'):
            return -self.g0**2/self.nk**2 * conv(G, self.gq2[:,:,None]*D, ['k-q,q','k-q,q'], [0,1], [True,True], self.beta)
        else:
            return -self.g0**2/self.nk**2 * conv(G, D, ['k-q,q','k-q,q'], [0,1], [True,True], self.beta)
    #------------------------------------------------------------
    def compute_GG(self, G):
        if hasattr(self, 'gq2'):
            return 2.0/self.nk**2 * self.gq2[:,:,None] * conv(G, -G[:,:,::-1], ['k,k+q','k,k+q'], [0,1], [True,True], self.beta)
        else:
            return 2.0/self.nk**2 * conv(G, -G[:,:,::-1], ['k,k+q','k,k+q'], [0,1], [True,True], self.beta)
    #------------------------------------------------------------
    def init_selfenergies(self):
        S  = zeros([self.nk,self.nk,self.ntau], dtype=complex)
        PI = zeros([self.nk,self.nk,self.ntau], dtype=complex)
        return S, PI
    #------------------------------------------------------------
    def compute_x0(self, F0x, D, jumpF0, jumpD):
        return -self.g0**2/self.nk**2 * conv(F0x, D, ['k-q,q','k-q,q', 'm,n-m'], [0,1,2], [True,True,False], self.beta, kinds=('fermion','boson','fermion'), op='...,...', jumps=(jumpF0, jumpD))

    def compute_gamma(self, F0gamma, D, jumpF0gamma, jumpD):
        return 1 - self.g0**2/self.nk**2 * conv(F0gamma, D, ['k-q,q','k-q,q', 'm,n-m'], [0,1,2], [True,True,False], self.beta, kinds=('fermion','boson','fermion'), op='...,...', jumps=(jumpF0gamma, jumpD))



if __name__=='__main__':
    def read_params(basedir, folder):
        
        params = {}
        for fname in os.listdir(os.path.join(basedir, 'data/', folder)):
            if '.npy' in fname:
                data = np.load(os.path.join(basedir, 'data/', folder, fname), allow_pickle=True)
                params[fname[:-4]] = data
    
        floats = ['beta', 'dens', 'dw', 'g0', 'mu', 'omega', \
                  'idelta', 't', 'tp', 'wmax', 'wmin']
        ints   = ['dim', 'nk', 'nw']
        bools  = ['sc']
        
        for f in floats:
            if f in params:
                params[f] = params[f][0]
        for i in ints:
            if i in ints:
                params[i] = int(params[i][0])
        for b in bools:
            if b in bools:
                params[b] = bool(params[b][0])
            
        params['band'] = None
    
        return params


    def plot():
        basedir = '../test/'
        folder = 'data_unrenormalized_nk12_abstp0.000_dim2_g00.16667_nw128_omega0.100_dens0.800_beta40.0000_QNone/'
        
        params = read_params(basedir, folder)
        G = np.load(os.path.join(basedir, 'data', folder, 'G.npy'))
        
        nk = params['nk']
        plt.figure()
        plt.plot(-1/np.pi * G[nk//4, nk//2, :].imag)
        plt.savefig('test')
        plt.close()
        
        A = -1/np.pi * G[:, nk//2, :].imag
        plt.figure()
        plt.imshow(A, origin='lower', aspect='auto')
        plt.colorbar()
        plt.savefig('test2')
        plt.close()
        
        
    plot()
    exit
    
    time0 = time.time()

    # example usage as follows :
    print('2D Renormalized Migdal')
 
    params = {}
    params['nw']    = 128
    params['nk']    = 12
    params['t']     = 0.040
    params['tp']    = 0.0
    params['omega'] = 0.100
    params['omega_q'] = omega_q(params['omega'], J=0.020, nk=params['nk'])
    params['dens']  = 0.8
    params['renormalized'] = False
    params['sc']    = 0
    params['band']  = band_square_lattice
    params['beta']  = 40.0
    params['dim']   = 2
    
    params['fixed_mu'] = 0

    #basedir = '/home/groups/simes/bln/data/elph/imagaxis/example/'
    basedir = '../test/'
    if not os.path.exists(basedir): os.makedirs(basedir)

    lamb = 1/6
    W    = 8.0
    params['g0'] = lamb2g0_ilya(lamb, params['omega'], W)
    print('g0 is ', params['g0'])
    
    #params['q0'] = None
    params['q0'] = 0.3
    if params['q0'] is not None:
        params['gq2'] = gexp_2d(params['nk'], q0=params['q0'])**2
    
    
    migdal = Migdal(params, basedir)

    sc_iter = 400
    S0, PI0, mu0  = None, None, None
    savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, mu0=None, frac=0.6)
    PI = params['g0']**2 * GG
    save(savedir + 'S.npy', S)
    save(savedir + 'PI.npy', PI)
    save(savedir + 'G.npy', G)
    save(savedir + 'D.npy', D)

    sc_iter = 100
    Xsc, Xcdw = migdal.susceptibilities(sc_iter, G, D, GG, frac=0.6)
    save(savedir + 'Xsc.npy',  [Xsc])
    save(savedir + 'Xcdw.npy', [Xcdw])

    print('------------------------------------------')
    print('simulation took', time.time()-time0, 's')


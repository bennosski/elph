from numpy import *
import numpy as np
from convolution import conv
from base import MigdalBase
from functions import mylamb2g0, band_1site, gexp_1d
import os
import time
import matplotlib.pyplot as plt


class Migdal(MigdalBase):
    #------------------------------------------------------------
    def setup(self):
        out = super().setup()
        assert self.dim==0
        return out
    #------------------------------------------------------------
    def compute_fill(self, Gw):
        return real(1.0 + 2./self.beta * 2.0*sum(Gw)) # second 2 accounting for negative matsubara freqs
    #------------------------------------------------------------
    def compute_n(self, Gtau):
        return -2.0*Gtau[-1].real
    
    
    def compute_n_tail(self, wn, ek, mu):
        # baresum is just nF
        baresum = 1 + np.tanh(-self.beta*(ek-mu)/2)
        bareGw = 1.0/(1j*wn - (ek-mu))
        return baresum - (1 + 2/self.beta * 2*np.sum(bareGw.real))

    
    #------------------------------------------------------------
    def compute_G(self, wn, ek, mu, S):
        return 1.0/(1j*wn - (ek-mu) - S)
    #------------------------------------------------------------
    def compute_D(self, vn, PI):
        return 1.0/(-(vn**2 + self.omega**2)/(2.0*self.omega) - PI)
    #------------------------------------------------------------
    def compute_S(self, G, D):
        return -self.g0**2 * G * D
    #------------------------------------------------------------
    def compute_GG(self, G):
        return 2.0 * G * (-G[::-1])
    #------------------------------------------------------------
    def init_selfenergies(self):
        S  = zeros(self.ntau, dtype=complex)
        PI = zeros(self.ntau, dtype=complex)
        return S, PI
 

if __name__=='__main__':
    time0 = time.time()

    # example usage as follows :
    print('1 site Migdal')

    params = {}
    params['nw']    = 2048
    params['nk']    = 0
    params['t']     = 1.0
    params['tp']    = 0.0
    params['omega'] = 1.0
    #params['dens']  = 0.9048
    #params['dens']  = 0.951
    params['dens']  = 1.058
    params['renormalized'] = False
    params['sc']    = False
    params['Q']     = None
    params['band']  = band_1site
    params['beta']  = 20.0
    params['dim']   = 0
    
    #basedir = '/home/groups/simes/bln/data/elph/imagaxis/example/'
    basedir = '/scratch/users/bln/elph/data/onesite/'
    if not os.path.exists(basedir): os.makedirs(basedir)

    params['g0'] = np.sqrt(5.5)
    print('g0 is ', params['g0'])
    
    migdal = Migdal(params, basedir)

    sc_iter = 2000
    S0, PI0, mu0  = None, None, None
    savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, mu0=None, frac=0.2)
    
    
    plt.figure()
    plt.plot(G)
    plt.title('Gtau')
    plt.ylim(-1, 0)

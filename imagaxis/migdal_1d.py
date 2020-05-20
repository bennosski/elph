from numpy import *
from convolution import conv
from base import MigdalBase
from functions import mylamb2g0, band_1d_lattice, gexp_1d
import os
import time

class Migdal(MigdalBase):
    #------------------------------------------------------------
    def setup(self):
        out = super().setup()
        assert self.dim==1
        return out
    #------------------------------------------------------------
    def compute_fill(self, Gw):
        return real(1.0 + 2./(self.nk * self.beta) * 2.0*sum(Gw)) # second 2 accounting for negative matsubara freqs
    #------------------------------------------------------------
    def compute_n(self, Gtau):
        return -2.0*mean(Gtau[:,-1]).real
    #------------------------------------------------------------
    def compute_G(self, wn, ek, mu, S):
        return 1.0/(1j*wn[None,:] - (ek[:,None]-mu) - S)
    #------------------------------------------------------------
    def compute_D(self, vn, PI):
        return 1.0/(-((vn**2)[None,:] + self.omega**2)/(2.0*self.omega) - PI)
    #------------------------------------------------------------
    def compute_S(self, G, D):
        return -self.g0**2/self.nk * conv(G, self.gq2[:,None]*D, ['k-q,q'], [0], [True], self.beta)
    #------------------------------------------------------------
    def compute_GG(self, G):
        return 2.0/self.nk * self.gq2[:,None] * conv(G, -G[:,::-1], ['k,k+q'], [0], [True], self.beta)
    #------------------------------------------------------------
    def init_selfenergies(self):
        S  = zeros([self.nk,self.ntau], dtype=complex)
        PI = zeros([self.nk,self.ntau], dtype=complex)
        return S, PI
    #------------------------------------------------------------
    def compute_x0(self, F0x, D, jumpF0, jumpD):
        return -self.g0**2/self.nk * conv(F0x, self.gq2[:,None]*D, ['k-q,q', 'm,n-m'], [0,1], [True,False], self.beta, kinds=('fermion','boson','fermion'), op='...,...', jumps=(jumpF0, jumpD))


if __name__=='__main__':
    time0 = time.time()

    # example usage as follows :
    print('1D Renormalized Migdal')

    '''
    params = {}
    params['nw']    = 512
    params['nk']    = 12
    params['t']     = 1.0
    params['tp']    = 0.0
    params['omega'] = 0.17
    params['dens']  = 0.8
    params['renormalized'] = False
    params['sc']    = False
    params['Q']     = None
    params['band']  = band_1d_lattice
    params['beta']  = 16.0
    params['dim']   = 1
    
    basedir = '/home/groups/simes/bln/data/elph/imagaxis/example/'
    if not os.path.exists(basedir): os.makedirs(basedir)

    lamb = 0.3
    W    = 8.0
    params['g0'] = mylamb2g0(lamb, params['omega'], W)
    params['q0'] = 0.1
    params['gq2'] = gexp_1d(params['nk'], q0=params['q0'])**2
    '''

    params = {}
    params['nw']    = 512
    params['nk']    = 200
    params['t']     = 1.0
    params['tp']    = 0.0
    params['omega'] = 1.0
    params['dens']  = 1.0
    params['renormalized'] = False
    params['sc']    = False
    params['Q']     = None
    params['band']  = band_1d_lattice
    params['beta']  = 20.0
    params['dim']   = 1
    
    basedir = '/home/groups/simes/bln/data/elph/imagaxis/example/'
    if not os.path.exists(basedir): os.makedirs(basedir)

    lamb = 0.1
    W    = 4.0
    params['g0'] = mylamb2g0(lamb, params['omega'], W)
    params['q0'] = None
    #params['gq2'] = gexp_1d(params['nk'], q0=params['q0'])**2
    params['gq2'] = ones(params['nk'])

    print('g0 is ', params['g0'])
    
    migdal = Migdal(params, basedir)

    sc_iter = 200
    S0, PI0, mu0  = None, None, None
    savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, mu0=None, frac=0.8)
    if not params['renormalized']:
        PI = None
    else:
        PI = params['g0']**2 * GG
    save(savedir + 'S.npy', S)
    save(savedir + 'PI.npy', PI)
    save(savedir + 'G.npy', G)
    save(savedir + 'D.npy', D)

    sc_iter = 20
    Xsc, Xcdw = migdal.susceptibilities(sc_iter, G, D, GG, frac=0.4)
    save(savedir + 'Xsc.npy',  [Xsc])
    save(savedir + 'Xcdw.npy', [Xcdw])

    print('------------------------------------------')
    print('simulation took', time.time()-time0, 's')


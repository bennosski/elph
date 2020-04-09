from numpy import *
from convolution import conv
from base import MigdalBase
from functions import lamb2g0_ilya, band_square_lattice
import os
import time

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
        return 1.0/(-((vn**2)[None,None,:] + self.omega**2)/(2.0*self.omega) - PI)
    #------------------------------------------------------------
    def compute_S(self, G, D):
        return -self.g0**2/self.nk**2 * conv(G, D, ['k-q,q','k-q,q'], [0,1], [True,True], self.beta)
    #------------------------------------------------------------
    def compute_GG(self, G):
        return 2.0/self.nk**2 * conv(G, -G[:,:,::-1], ['k,k+q','k,k+q'], [0,1], [True,True], self.beta)
    #------------------------------------------------------------
    def init_selfenergies(self):
        S  = zeros([self.nk,self.nk,self.ntau], dtype=complex)
        PI = zeros([self.nk,self.nk,self.ntau], dtype=complex)
        return S, PI
    #------------------------------------------------------------
    def compute_x0(self, F0x, D, jumpF0, jumpD):
        return -self.g0**2/self.nk**2 * conv(F0x, D, ['k-q,q','k-q,q', 'm,n-m'], [0,1,2], [True,True,False], self.beta, kinds=('fermion','boson','fermion'), op='...,...', jumps=(jumpF0, jumpD))


if __name__=='__main__':
    time0 = time.time()

    # example usage as follows :
    print('2D Renormalized Migdal')
 
    params = {}
    params['nw']    = 512
    params['nk']    = 8
    params['t']     = 1.0
    params['tp']    = -0.3
    params['omega'] = 0.17
    params['dens']  = 0.8
    params['renormalized'] = True
    params['sc']    = 0
    params['Q']     = None
    params['band']  = band_square_lattice
    params['beta']  = 6.0
    params['g0']    = 0.125
    params['dim']   = 2


    #basedir = '/home/groups/simes/bln/data/elph/imagaxis/example/'
    basedir = 'test/'
    if not os.path.exists(basedir): os.makedirs(basedir)

    lamb = 0.1
    W    = 8.0
    params['g0'] = lamb2g0_ilya(lamb, params['omega'], W)
    print('g0 is ', params['g0'])
    
    migdal = Migdal(params, basedir)

    sc_iter = 10
    S0, PI0, mu0  = None, None, None
    savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, mu0=None, frac=0.2)
    PI = params['g0']**2 * GG
    save(savedir + 'S.npy', S)
    save(savedir + 'PI.npy', PI)
    save(savedir + 'G.npy', G)
    save(savedir + 'D.npy', D)

    sc_iter = 100
    Xsc, Xcdw = migdal.susceptibilities(sc_iter, G, D, GG, frac=0.2)
    save(savedir + 'Xsc.npy',  [Xsc])
    save(savedir + 'Xcdw.npy', [Xcdw])

    print('------------------------------------------')
    print('simulation took', time.time()-time0, 's')


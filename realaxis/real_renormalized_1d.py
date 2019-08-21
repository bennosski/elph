import imagaxis
from renormalized_1d import Migdal
from convolution import conv
from functions import band_1dsquare_lattice, mylamb2g0
import os

class RealAxisMigdal(Migdal):
   
    def __init__(self, params, basedir):
        super().__init__(params, basedir)

    # todo
    # functions for real axis Green's function
 
    def compute_G(self):
        pass

    def compute_D(self):
        pass

    def compute_S(self):
        pass

    def compute_PI(self):
        pass

    
    def selfconsistency(self, sc_iter, frac=0.5, alpha=0.5, S0=None, PI0=None):
        savedir, G, D, S, GG =  super().selfconsistency(sc_iter, frac=frac, alpha=alpha, S0=S0, PI0=PI0)

        # now next steps

        # compute Gsum
        

        # selfconsistency loop...

        

"""
def compute_PI_real_axis(GR, Gsum):
    GA = conj(GR)
    A  = -1.0/pi * GR.imag
    return 2.0*g**2*dw/Nk**2*(conv(A, Gsum, ['k+q,k','k+q,k','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
                    -conv(A, GA*nF[None,None,:], ['k+q,k','k+q,k','w+z,z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
                    +conv(A*nF[None,None,:], GA, ['k+q,k','k+q,k','w+z,z'], [0,1,2], [True,True,False])[:,:,:len(w)])

def compute_S_real_axis(GR, DR, Gsum):
    # compute selfenergy from Marsiglio formula
    B  = -1.0/pi * DR.imag
    return -g**2*dw/Nk**2*(conv(B, Gsum, ['k-q,q','k-q,q','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
             -conv(B*(1+nB)[None,None,:], GR, ['k-q,q','k-q,q','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)] \
             +conv(B, GR*nF[None,None,:], ['k-q,q','k-q,q','z,w-z'], [0,1,2], [True,True,False])[:,:,:len(w)])

    def test(self):
        print(self.nk)
        print(self.nw)
"""


if __name__=='__main__':
    print('1D Renormalized Migdal Real Axis')

    params = {}
    params['nw']    = 512
    params['nk']    = 12
    params['t']     = 1.0
    params['tp']    = 0.0
    params['omega'] = 0.17
    params['dens']  = 0.8
    params['renormalized'] = True
    params['sc']    = 0
    params['band']  = band_1dsquare_lattice
    params['beta']  = 16.0
    params['g0']    = 0.125
    params['dim']   = 1

    basedir = '/home/groups/simes/bln/data/elph/imagaxis/example/'
    if not os.path.exists(basedir): os.makedirs(basedir)

    lamb = 0.1
    W    = 8.0
    params['g0'] = mylamb2g0(lamb, params['omega'], W)
    print('g0 is ', params['g0'])
    
    migdal = Migdal(params, basedir)

    sc_iter, S0, PI0 = 10, None, None
    migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, mu0=None)

    

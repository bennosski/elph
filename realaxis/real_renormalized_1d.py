import imagaxis
from renormalized_2d import Migdal
from convolution import conv

class RealAxisMigdal(Migdal):
   
    def __init__(self, params, basedir):
        super().__init__(params, basedir)

    # todo
    # test
    # functions for real axis Green's function
 
    def compute_GR(self):
        pass

    def compute_DR(self):
        pass

    def compute_S_real_axsi(self):
        pass

    def compute_PI_real_axis(self):
        pass


    
    def selfconsistency_realaxis(self, sc_iter, frac=0.9, alpha=0.5, S0=None, PI0=None):
        savedir, G, D, S, GG =  super.selfconsistency(sc_iter, frac=frac, alpha=alpha, S0=S0, PI0=PI0)

        # now next steps

        # compute Gsum

        # selfconsistency loop...


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

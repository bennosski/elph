import imagaxis
from migdal_1site import Migdal
from convolution import basic_conv
from functions import band_1site, mylamb2g0, gexp_1d
import os
import numpy as np
import fourier
from anderson import AndersonMixing
import matplotlib.pyplot as plt

class RealAxisMigdal(Migdal):
    #-----------------------------------------------------------
    def __init__(self, params, basedir):
        super().__init__(params, basedir)
        
    #-----------------------------------------------------------
    def setup_realaxis(self):
        w = np.arange(self.wmin, self.wmax, self.dw)
        self.nr = len(w)
        assert self.nr%2==0 and abs(w[self.nr//2])<1e-10

        nB = 1.0/(np.exp(self.beta*w)-1.0)
        nF = 1.0/(np.exp(self.beta*w)+1.0)
        DRbareinv = ((w+self.idelta)**2 - self.omega**2)/(2.0*self.omega)

        wn = (2*np.arange(self.nw)+1) * np.pi / self.beta
        vn = (2*np.arange(self.nw+1)) * np.pi / self.beta
        ek = self.band(self.nk, 1.0, self.tp)
        
        return wn, vn, ek, w, nB, nF, DRbareinv
    #----------------------------------------------------------- 
    def compute_GR(self, w, ek, mu, SR):
        return 1.0/(w + self.idelta - (ek-mu) - SR)
    
    #-----------------------------------------------------------
    def compute_DR(self, DRbareinv, PIR):
        return 1.0/(DRbareinv - PIR)
    
    #-----------------------------------------------------------
    def compute_SR(self, GR, Gsum, DR, nB, nF):
        B = -1.0/np.pi * DR.imag
        return -self.g0**2*self.dw*(basic_conv(B, Gsum, ['z,w-z'], [0], [False])[:self.nr] \
             -basic_conv(B*(1+nB), GR, ['z,w-z'], [0], [False])[:self.nr] \
             +basic_conv(B, GR*nF, ['z,w-z'], [0], [False])[:self.nr])

    #-----------------------------------------------------------
    def compute_PIR(self, GR, Gsum, nF):
        GA = np.conj(GR)
        A  = -1.0/np.pi * GR.imag
        return 2.0*self.g0**2*self.dw*(basic_conv(A, Gsum, ['z,w-z'], [0], [False])[:self.nr] \
                        -basic_conv(A, GA*nF, ['w+z,z'], [0], [False])[:self.nr] \
                        +basic_conv(A*nF, GA, ['w+z,z'], [0], [False])[:self.nr])
    
    #-----------------------------------------------------------    
    def selfconsistency(self, sc_iter, frac=0.5, alpha=0.5, S0=None, PI0=None, mu0=None):

        savedir, mu, G, D, S, GG = super().selfconsistency(sc_iter, frac=frac, cont=True)

        print('\nReal-axis selfconsistency')
        print('---------------------------------')
        
        # imag axis failed to converge
        if savedir is None: exit()

        wn, vn, ek, w, nB, nF, DRbareinv = self.setup_realaxis()
        
        SR  = np.zeros(self.nr, dtype=complex)
        PIR = np.zeros(self.nr, dtype=complex)
        GR  = self.compute_GR(w, ek, mu, SR)
        DR  = self.compute_DR(DRbareinv, PIR)
        
        # convert to imaginary frequency
        G = fourier.t2w(G, self.beta, self.dim, 'fermion')[0]

        # compute Gsum
        if self.renormalized:
            Gsum_plus  = np.zeros(self.nr, dtype=complex)
        Gsum_minus = np.zeros(self.nr, dtype=complex)
        for i in range(self.nr):
            if self.renormalized:
                Gsum_plus[i]  = np.sum(G/((w[i]+1j*wn)), axis=0) / self.beta 
            Gsum_minus[i] = np.sum(G/((w[i]-1j*wn)), axis=0) / self.beta
        # handle sum over pos and negative freqs
        if self.renormalized:
            Gsum_plus  += np.conj(Gsum_plus)
        Gsum_minus += np.conj(Gsum_minus)
        print('finished Gsum')    

        # can I always use Gsum_plus[::-1]? what if w's are not symmetric?

        AMSR  = AndersonMixing(alpha=alpha)
        AMPIR = AndersonMixing(alpha=alpha)
        
        # selfconsistency loop
        change = [0,0]
        for i in range(sc_iter):
            SR0 = SR[:]
            PIR0 = PIR[:]

            SR  = self.compute_SR(GR, Gsum_minus, DR, nB, nF)
            #SR  = AMSR.step(SR0, SR)
            SR  = frac*SR  + (1.0-frac)*SR0
            GR = self.compute_GR(w, ek, mu, SR)
            change[0] = np.mean(np.abs(SR-SR0))/np.mean(np.abs(SR+SR0))

            if self.renormalized:
                PIR = self.compute_PIR(GR, Gsum_plus, nF)            
                #PIR = AMPIR.step(PIR0, PIR)
                PIR = frac*PIR + (1.0-frac)*PIR0
                DR = self.compute_DR(DRbareinv, PIR)
                change[1] = np.mean(np.abs(PIR-PIR0))/np.mean(np.abs(PIR+PIR0))

            
            if i%1==0: print('change = %1.3e, %1.3e'%(change[0], change[1]))
    
            if i>5 and np.sum(change)<2e-15: break

        np.save('savedir.npy', [savedir])            
        np.save(savedir+'w', w)
        np.save(savedir+'GR', GR)
        np.save(savedir+'SR', SR)
        np.save(savedir+'DR', DR)
        np.save(savedir+'PIR', PIR)
        
        return savedir, w, GR, DR, SR, PIR
            

if __name__=='__main__':
    print('1D Renormalized Migdal Real Axis')

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
    
    
    basedir = '/scratch/users/bln/elph/data/onesite/'

    params['dw']     = 0.0004
    params['wmin']   = -10
    params['wmax']   = +10
    params['idelta'] = 0.001j
    

    params['g0'] = np.sqrt(5.5)
    
    migdal = RealAxisMigdal(params, basedir)

    sc_iter, S0, PI0 = 200, None, None
    savedir, w, GR, DR, SR, PIR = migdal.selfconsistency(sc_iter, frac=0.4)

    A = -1/np.pi * GR.imag
 
    plt.figure()
    plt.plot(w, A)
    
    
    nF = 1/(np.exp(w*params['beta']) + 1)
    dw = (w[-1]-w[0])/(len(w)-1)
    print('norm', np.trapz(A, dx=dw))
    print('filling from A(k,w) : ', 2*np.trapz(A*nF, dx=dw))
    
    
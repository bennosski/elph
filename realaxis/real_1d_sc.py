import imagaxis
from migdal_1d_sc import Migdal
from convolution import basic_conv
from functions import band_1d_lattice, mylamb2g0
import os
import numpy as np
import fourier
from anderson import AndersonMixing


class RealAxisMigdal(Migdal):
    #-----------------------------------------------------------
    def __init__(self, params, basedir):
        super().__init__(params, basedir)
        
    #-----------------------------------------------------------
    def setup_realaxis(self):
        w = np.arange(self.wmin, self.wmax, self.dw)
        self.nr = len(w)
        assert self.nr%2==0 and abs(w[self.nr//2])<1e-10
        #assert self.sc is True

        nB = 1.0/(np.exp(self.beta*w)-1.0)
        nF = 1.0/(np.exp(self.beta*w)+1.0)
        DRbareinv = ((w+self.idelta)**2 - self.omega**2)/(2.0*self.omega)

        wn = (2*np.arange(self.nw)+1) * np.pi / self.beta
        vn = (2*np.arange(self.nw+1)) * np.pi / self.beta
        ek = self.band(self.nk, 1.0, self.tp)
        
        return wn, vn, ek, w, nB, nF, DRbareinv
    #----------------------------------------------------------- 
    def compute_GR(self, w, ek, mu, SR):
        return np.linalg.inv((w[None,:,None,None]+self.idelta)*Migdal.tau0[None,None,:,:] \
                  - (ek[:,None,None,None]-mu)*Migdal.tau3[None,None,:,:] \
                  - SR)
    
    #-----------------------------------------------------------
    def compute_DR(self, DRbareinv, PIR):
        return 1.0/(DRbareinv[None,:] - PIR)
    
    #-----------------------------------------------------------
    def compute_SR(self, GR, tau3Gsumtau3, DR, nB, nF):
        B  = -1.0/np.pi * DR.imag[:,:,None,None] * np.ones([self.nk,self.nr,2,2])
        tau3GRtau3 = np.einsum('ab,...bc,cd->...ad',Migdal.tau3,GR,Migdal.tau3)
        return -self.g0**2*self.dw/self.nk*( \
                basic_conv(B, tau3Gsumtau3, ['q,k-q','z,w-z'], [0,1], [True,False])[:,:self.nr] \
               -basic_conv(B*(1+nB[None,:,None,None]), tau3GRtau3, ['q,k-q','z,w-z'], [0,1], [True,False])[:,:self.nr] \
               +basic_conv(B, tau3GRtau3*nF[None,:,None,None], ['q,k-q','z,w-z'], [0,1], [True,False])[:,:self.nr])

    #-----------------------------------------------------------
    def compute_PIR(self, GR, tau3Gsumtau3, nF):
        tau3GAtau3 = np.einsum('ab,...bc,cd->...ad', Migdal.tau3, np.conj(GR), Migdal.tau3)
        A  = -1.0/np.pi * GR.imag
        return self.g0**2*self.dw/self.nk * np.einsum('...aa->...', \
                basic_conv(A, tau3Gsumtau3, ['k+q,k','z,w-z'], [0,1], [True,False], op='...ab,...bc->...ac')[:,:self.nr] \
               -basic_conv(A, tau3GAtau3*nF[None,:,None,None], ['k+q,k','w+z,z'], [0,1], [True,False], op='...ab,...bc->...ac')[:,:self.nr] \
               +basic_conv(A*nF[None,:,None,None], tau3GAtau3, ['k+q,k','w+z,z'], [0,1], [True,False], op='...ab,...bc->...ac')[:,:self.nr])
    
    #-----------------------------------------------------------    
    def selfconsistency(self, sc_iter, frac=0.5, alpha=0.5, S0=None, PI0=None, mu0=None):
        savedir, mu, G, D, S, GG = super().selfconsistency(sc_iter, frac=frac, alpha=alpha, S0=S0, PI0=PI0, mu0=mu0)

        np.save(savedir+'S.npy', S)
        #np.save(savedir+'PI.npy', self.g0**2 * GG)

        print('\nReal-axis selfconsistency')
        print('---------------------------------')
        
        # imag axis failed to converge
        if savedir is None: exit()

        wn, vn, ek, w, nB, nF, DRbareinv = self.setup_realaxis()
        
        SR  = np.zeros([self.nk,self.nr,2,2], dtype=complex)
        PIR = np.zeros([self.nk,self.nr], dtype=complex)
        GR  = self.compute_GR(w, ek, mu, SR)
        DR  = self.compute_DR(DRbareinv, PIR)
        
        # convert to imaginary frequency
        G = fourier.t2w(G, self.beta, self.dim, 'fermion')[0]

        # compute Gsum
        if self.renormalized:
            Gsum_plus  = np.zeros([self.nk,self.nr,2,2], dtype=complex)
        Gsum_minus = np.zeros([self.nk,self.nr,2,2], dtype=complex)
        for i in range(self.nr):
            if self.renormalized:
                Gsum_plus[:,i]  = np.sum(G/((w[i]+1j*wn)[None,:,None,None]), axis=1) / self.beta 
            Gsum_minus[:,i] = np.sum(G/((w[i]-1j*wn)[None,:,None,None]), axis=1) / self.beta
        # handle sum over pos and negative freqs
        if self.renormalized:
            Gsum_plus  += np.conj(Gsum_plus)
            Gsum_plus = np.einsum('ab,...bc,cd->...ad', Migdal.tau3, Gsum_plus, Migdal.tau3)
        Gsum_minus += np.conj(Gsum_minus)
        Gsum_minus = np.einsum('ab,...bc,cd->...ad', Migdal.tau3, Gsum_minus, Migdal.tau3)
        print('finished Gsum')    

        # can I always use Gsum_plus[::-1]? what if w's are not symmetric?

        AMSR  = AndersonMixing(alpha=alpha)
        AMPIR = AndersonMixing(alpha=alpha)
        
        # selfconsistency loop
        change = [0,0]
        for i in range(5):
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
            

if __name__=='__main__':
    print('1D Renormalized Migdal Real Axis')

    params = {}
    params['nw']    = 512
    params['nk']    = 200
    params['t']     = 1.0
    params['tp']    = 0.0
    params['omega'] = 0.5
    params['dens']  = 1.0
    params['renormalized'] = True
    params['sc'] = True
    params['band']  = band_1d_lattice
    params['beta']  = 60.0
    params['dim']   = 1
    lamb = 1.0
    W    = 4.0
    params['g0'] = mylamb2g0(lamb, params['omega'], W)
    params['gq2'] = 1

    params['dw']     = 0.001
    params['wmin']   = -4.1
    params['wmax']   = +4.1
    params['idelta'] = 0.010j
    
    basedir = '/home/groups/simes/bln/data/elph/imagaxis/example/'
    if not os.path.exists(basedir): os.makedirs(basedir)
    
    migdal = RealAxisMigdal(params, basedir)

    if not params['renormalized']:
        sc_iter, S0, PI0 = 2000, None, None
        migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, mu0=None, frac=0.2, alpha=None)

    if params['renormalized']: 
        sc_iter, S0, PI0 = 2000, None, None

        loaddir = basedir + 'data/data_unrenormalized_nk200_abstp0.000_dim1_g01.00000_nw512_omega0.500_dens1.000_beta60.0000_sc1/' 
        S0  = np.load(loaddir+'S.npy')
        PI0 = np.zeros((params['nk'], 2*params['nw']+1), dtype=complex)

        migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, mu0=None, frac=0.02, alpha=None)

   

    

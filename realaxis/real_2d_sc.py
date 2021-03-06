import imagaxis
from migdal_2d_sc import Migdal
from convolution import basic_conv
from functions import band_square_lattice, mylamb2g0
import os
import numpy as np
import fourier
from anderson import AndersonMixing
import time


class RealAxisMigdal(Migdal):
    #-----------------------------------------------------------
    def __init__(self, params, basedir):
        super().__init__(params, basedir)
        
    #-----------------------------------------------------------
    def setup_realaxis(self):
        w = np.arange(self.wmin, self.wmax + self.dw/2, self.dw)
        self.nr = len(w)
        self.izero = np.argmin(np.abs(w))
        assert abs(w[self.izero])<1e-10
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
        return np.linalg.inv((w[None,None,:,None,None]+self.idelta)*Migdal.tau0[None,None,None,:,:] \
                  - (ek[:,:,None,None,None]-mu)*Migdal.tau3[None,None,None,:,:] \
                  - SR)
    
    #-----------------------------------------------------------
    def compute_DR(self, DRbareinv, PIR):
        return 1.0/(DRbareinv[None,None,:] - PIR)
    
    #-----------------------------------------------------------
    def compute_SR(self, GR, tau3Gsumtau3, DR, nB, nF):
        B  = -1.0/np.pi * DR.imag[:,:,:,None,None] * np.ones([self.nk,self.nk,self.nr,2,2])        
        tau3GRtau3 = np.einsum('ab,...bc,cd->...ad',Migdal.tau3,GR,Migdal.tau3)

        izeros = [self.nk//2, self.nk//2, self.izero]

        def conv(a, b):
            return basic_conv(a, b, ['q,k-q','q,k-q','z,w-z'], [0,1,2], [True,True,False], izeros=izeros)[:,:,:self.nr]

        return -self.g0**2*self.dw/self.nk**2*(conv(B, tau3Gsumtau3) - conv(B*(1+nB)[None,None,:,None,None], tau3GRtau3) + conv(B, tau3GRtau3*nF[None,None,:,None,None]))

        
    #-----------------------------------------------------------
    def compute_PIR(self, GR, tau3Gsumtau3, nF):
        tau3GAtau3 = np.einsum('ab,...bc,cd->...ad', Migdal.tau3, np.conj(GR), Migdal.tau3)
        A  = -1.0/np.pi * GR.imag

        izeros = [self.nk//2, self.nk//2, self.izero]
         
        def conv(a, b, fc):
            return basic_conv(a, b, ['k+q,k','k+q,k',fc], [0,1,2], [True,True,False], izeros=izeros, op='...ab,...bc->...ac')[:,:,:self.nr]

        return self.g0**2*self.dw/self.nk**2*np.einsum('...aa->...', conv(A, tau3Gsumtau3, 'z,w-z') - conv(A, tau3GAtau3*nF[None,None,:,None,None], 'w+z,z') + conv(A*nF[None,None,:,None,None], tau3GAtau3, 'w+z,z'))

 
        '''
        return self.g0**2*self.dw/self.nk**2 * np.einsum('...aa->...', \
           basic_conv(tau3A, Gsum, ['k+q,k','k+q,k','z,w-z'], [0,1,2], [True,True,False], op='...ab,...bc->...ac')[:,:,:self.nr] \
          -basic_conv(tau3A, tau3GA*nF[None,None,:,None,None], ['k+q,k','k+q,k','w+z,z'], [0,1,2], [True,True,False], op='...ab,...bc->...ac')[:,:,:self.nr] \
          +basic_conv(tau3A*nF[None,None,:,None,None], tau3GA, ['k+q,k','k+q,k','w+z,z'], [0,1,2], [True,True,False], op='...ab,...bc->...ac')[:,:,:self.nr])
        '''

    #-----------------------------------------------------------    
    def selfconsistency(self, sc_iter, frac=0.5, fracR=0.5, alpha=0.5, S0=None, PI0=None, mu0=None, cont=False, interp=None):
        
        savedir, mu, G, D, S, GG = super().selfconsistency(sc_iter, frac=frac, alpha=alpha, S0=S0, PI0=PI0, mu0=mu0, cont=False)

        # imag axis failed to converge
        if savedir is None: exit()

        savedir = savedir[:-1] + '_idelta{:.4f}_w{:.4f}_{:.4f}/'.format(self.idelta.imag, np.abs(self.wmin), self.wmax)
        if not os.path.exists(savedir): os.makedirs(savedir)

        for key in self.keys:
            np.save(savedir+key, [getattr(self, key)])

        np.save(savedir+'mu', [mu])

        print('savedir ', savedir)


        del D
        del GG
        del S


        print('\nReal-axis selfconsistency')
        print('---------------------------------')
        

        wn, vn, ek, w, nB, nF, DRbareinv = self.setup_realaxis()

        if interp is not None:
            SR = interp.SR
            PIR = interp.PIR
        if os.path.exists(savedir+'SR.npy'):
            if cont:
                print('CONTINUING FROM EXISTING REAL AXIS DATA')
                SR = np.load(savedir+'SR.npy')
                PIR = np.load(savedir+'PIR.npy')
            else:
                print('data exists. Not continuing. Set cont=True or delete data.')
                exit()
        
        else:
            SR  = np.zeros([self.nk,self.nk,self.nr,2,2], dtype=complex)
            PIR = np.zeros([self.nk,self.nk,self.nr], dtype=complex)
        
        GR  = self.compute_GR(w, ek, mu, SR)
        DR  = self.compute_DR(DRbareinv, PIR)
        
        # convert to imaginary frequency
        G = fourier.t2w(G, self.beta, self.dim, 'fermion')[0]

        # compute Gsum
        if self.renormalized:
            tau3Gsum_plustau3  = np.zeros([self.nk,self.nk,self.nr,2,2], dtype=complex)
        tau3Gsum_minustau3 = np.zeros([self.nk,self.nk,self.nr,2,2], dtype=complex)
        for i in range(self.nr):
            if self.renormalized:
                tau3Gsum_plustau3[:,:,i]  = np.sum(G/((w[i]+1j*wn)[None,None,:,None,None]), axis=2) / self.beta 
            tau3Gsum_minustau3[:,:,i] = np.sum(G/((w[i]-1j*wn)[None,None,:,None,None]), axis=2) / self.beta
        # handle sum over pos and negative freqs
        if self.renormalized:
            tau3Gsum_plustau3  += np.conj(tau3Gsum_plustau3)
            tau3Gsum_plustau3 = np.einsum('ab,...bc,cd->...ad', Migdal.tau3, tau3Gsum_plustau3, Migdal.tau3)

        tau3Gsum_minustau3 += np.conj(tau3Gsum_minustau3)
        tau3Gsum_minustau3 = np.einsum('ab,...bc,cd->...ad', Migdal.tau3, tau3Gsum_minustau3, Migdal.tau3)

        print('finished Gsum')    

        np.save(os.path.join(savedir, 'tau3Gsum_minustau3.npy'), tau3Gsum_minustau3)
        if self.renormalized:
            np.save(os.path.join(savedir, 'tau3Gsum_plustau3.npy'), tau3Gsum_plustau3)

        del G

        # can I always use Gsum_plus[::-1]? what if w's are not symmetric?

        #AMSR  = AndersonMixing(alpha=alpha)
        #AMPIR = AndersonMixing(alpha=alpha)
        
        # selfconsistency loop
        change = [0,0]
        best_chg = None
        for i in range(sc_iter):
            SR0 = SR[:]
            PIR0 = PIR[:]

            SR  = self.compute_SR(GR, tau3Gsum_minustau3, DR, nB, nF)
            #SR  = AMSR.step(SR0, SR)
            SR  = fracR*SR  + (1.0-fracR)*SR0
            GR = self.compute_GR(w, ek, mu, SR)
            change[0] = np.mean(np.abs(SR-SR0))/np.mean(np.abs(SR+SR0))

            if self.renormalized:
                PIR = self.compute_PIR(GR, tau3Gsum_plustau3, nF)            
                #PIR = AMPIR.step(PIR0, PIR)
                PIR = fracR*PIR + (1.0-fracR)*PIR0
                DR = self.compute_DR(DRbareinv, PIR)
                change[1] = np.mean(np.abs(PIR-PIR0))/np.mean(np.abs(PIR+PIR0))

            if i%1==0: print('change = %1.3e, %1.3e'%(change[0], change[1]))
    
            if best_chg is None or np.mean(change) < best_chg:
                best_chg = np.mean(change)
            
                np.save(savedir+'savedir.npy', [savedir])            
                np.save(savedir+'realchg.npy', [np.mean(change)])
                np.save(savedir+'w', w)
                np.save(savedir+'GR', GR)
                np.save(savedir+'SR', SR)
                np.save(savedir+'DR', DR)
                np.save(savedir+'PIR', PIR)

                np.save(savedir+'GRbackup', GR)
                np.save(savedir+'SRbackup', SR)
                np.save(savedir+'DRbackup', DR)
                np.save(savedir+'PIRbackup', PIR)
    
            if i>5 and np.sum(change)<2e-15: break
        

if __name__=='__main__':
    time0 = time.time()

    print('2D Renormalized Migdal Real Axis')

    params = {}
    params['nw']    = 512
    params['nk']    = 60
    params['t']     = 1.0

    if sys.argv[4]=='False':
        params['tp']    = -0.3
        params['dens']  = 0.8
    elif sys.argv[4]=='True':
        params['tp'] = 0.0
        params['dens'] = 1.0
    else:
        raise Exception

    params['omega'] = 1.0
    params['sc'] = True
    params['band']  = band_square_lattice
    params['dim']   = 2

    #ibeta = int(sys.argv[1])
    #betas = np.linspace(1.0, 50.0, 50) 
    #params['beta']  = betas[ibeta] 
    
    params['beta']  = float(sys.argv[1]) 
    renorm = True if sys.argv[2]=='True' else False
    params['renormalized'] = renorm
    #lamb = 0.2
    #lamb = 0.3   # try this one as well......
    lamb = float(sys.argv[3])

    W    = 8.0
    params['g0'] = mylamb2g0(lamb, params['omega'], W)
    params['gq2'] = 1

    params['dw']     = 0.001
    params['wmin']   = -4.1
    params['wmax']   = +4.1
    params['idelta'] = 0.010j
    
    basedir = '/home/groups/simes/bln/data/elph/production/'
    if not os.path.exists(basedir): os.makedirs(basedir)
    
    migdal = RealAxisMigdal(params, basedir)

    sc_iter, S0, PI0 = 2000, None, None
    migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, mu0=None, frac=0.2, alpha=None)

    print('\ntotal program took {}s'.format(time.time()-time0))
   

    

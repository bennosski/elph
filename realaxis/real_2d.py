import imagaxis
from migdal_2d import Migdal
from convolution import basic_conv
from functions import band_square_lattice, mylamb2g0, gexp_2d, omega_q
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
        
        if hasattr(self, 'omega_q'):
            DRbareinv = ((w[None,None,:]+self.idelta)**2 - self.omega_q[:,:,None]**2)/(2.0*self.omega_q[:,:,None])
        else:
            DRbareinv = ((w+self.idelta)**2 - self.omega**2)/(2.0*self.omega)
            
        wn = (2*np.arange(self.nw)+1) * np.pi / self.beta
        vn = (2*np.arange(self.nw+1)) * np.pi / self.beta
        ek = self.band(self.nk, self.t, self.tp)
        #ek = band_square_lattice(self.nk, self.t, self.tp)        

        return wn, vn, ek, w, nB, nF, DRbareinv
    #----------------------------------------------------------- 
    def compute_GR(self, w, ek, mu, SR):
        return 1.0/(w[None,None,:]+self.idelta - (ek[:,:,None]-mu) - SR)
    
    #-----------------------------------------------------------
    def compute_DR(self, DRbareinv, PIR):
        if hasattr(self, 'omega_q'):
            return 1.0/(DRbareinv - PIR)
        else:
            return 1.0/(DRbareinv[None,None,:] - PIR)
    
    #-----------------------------------------------------------
    def compute_SR(self, GR, Gsum, DR, nB, nF):
        B = -1.0/np.pi * DR.imag
        if hasattr(self, 'gq2'):
            return -self.g0**2*self.dw/self.nk**2*(basic_conv(self.gq2[:,:,None]*B, Gsum, ['q,k-q','q,k-q','z,w-z'], [0,1,2], [True,True,False])[:,:,:self.nr]                
               -basic_conv(self.gq2[:,:,None]*B*(1+nB)[None,None,:], GR, ['q,k-q','q,k-q','z,w-z'], [0,1,2], [True,True,False])[:,:,:self.nr] \
               +basic_conv(self.gq2[:,:,None]*B, GR*nF[None,None,:], ['q,k-q','q,k-q','z,w-z'], [0,1,2], [True,True,False])[:,:,:self.nr])        
        else:
            return -self.g0**2*self.dw/self.nk**2*(basic_conv(B, Gsum, ['q,k-q','q,k-q','z,w-z'], [0,1,2], [True,True,False])[:,:,:self.nr]                
               -basic_conv(B*(1+nB)[None,None,:], GR, ['q,k-q','q,k-q','z,w-z'], [0,1,2], [True,True,False])[:,:,:self.nr] \
               +basic_conv(B, GR*nF[None,None,:], ['q,k-q','q,k-q','z,w-z'], [0,1,2], [True,True,False])[:,:,:self.nr])

    #-----------------------------------------------------------
    def compute_PIR(self, GR, Gsum, nF):
        GA = np.conj(GR)
        A  = -1.0/np.pi * GR.imag
        if hasattr(self, 'gq2'):
            return 2.0*self.g0**2*self.gq2[:,:,None]*self.dw/self.nk**2*(basic_conv(A, Gsum, ['k+q,k','k+q,k','z,w-z'], [0,1,2], [True,True,False])[:,:,:self.nr] \
               -basic_conv(A, GA*nF[None,None,:], ['k+q,k','k+q,k','w+z,z'], [0,1,2], [True,True,False])[:,:,:self.nr] \
               +basic_conv(A*nF[None,None,:], GA, ['k+q,k','k+q,k','w+z,z'], [0,1,2], [True,True,False])[:,:,:self.nr])
        else:
            return 2.0*self.g0**2*self.dw/self.nk**2*(basic_conv(A, Gsum, ['k+q,k','k+q,k','z,w-z'], [0,1,2], [True,True,False])[:,:,:self.nr] \
               -basic_conv(A, GA*nF[None,None,:], ['k+q,k','k+q,k','w+z,z'], [0,1,2], [True,True,False])[:,:,:self.nr] \
               +basic_conv(A*nF[None,None,:], GA, ['k+q,k','k+q,k','w+z,z'], [0,1,2], [True,True,False])[:,:,:self.nr])


    def compute_jjcorr(self, w, wn, G, GR, nF):
        # convert to imaginary frequency
        G = fourier.t2w(G, self.beta, self.dim, 'fermion')[0]

        # compute Gsum
        Gsum  = np.zeros([self.nk,self.nk,self.nr], dtype=complex)
        for i in range(self.nr):
            Gsum[:,:,i]  = np.sum(G/((w[i]+1j*wn)[None,None,:]), axis=2) / self.beta 
        # handle sum over pos and negative freqs
        Gsum  += np.conj(Gsum)
        print('finished Gsum')    

        del G

        ks = np.arange(-np.pi, np.pi, 2*np.pi/self.nk)
        sin = np.sin(ks)
        cos = np.cos(ks)
        fk = (-2*self.t*sin[:,None] - 4*self.tp*sin[:,None]*cos[None,:])
        GA = np.conj(GR)
        A  = -1.0/np.pi * GR.imag

        A *= (fk**2)[:,:,None]
        jj0w = 2.0 * self.dw / self.nk**2 * np.sum(
                basic_conv(A, Gsum, ['z,w-z'], [2], [False])[:,:,:self.nr] \
               -basic_conv(A, GA*nF[None,None,:], ['w+z,z'], [2], [False])[:,:,:self.nr] \
               +basic_conv(A*nF[None,None,:], GA, ['w+z,z'], [2], [False])[:,:,:self.nr], axis=(0,1))

        np.save('jj0w{}'.format('r' if self.renormalized else 'u'), jj0w)

        del GA
        del A
        del Gsum

        print('done jj0w')

        jjw = jj0w / (1 + self.g0**2 * (-2/self.omega) * jj0w)
        np.save('jjw{}'.format('r' if self.renormalized else 'u'), jjw)

    
    #-----------------------------------------------------------    
    def selfconsistency(self, sc_iter, frac=0.5, alpha=0.5, S0=None, PI0=None, mu0=None, cont=False):
        savedir, mu, G, D, S, GG = super().selfconsistency(sc_iter, frac=frac, alpha=alpha, S0=S0, PI0=PI0, mu0=mu0)

        for key in self.keys:
            np.save(savedir+key, [getattr(self, key)])

        print('savedir ', savedir)

        del D
        del GG
        del S

        print('\nReal-axis selfconsistency')
        print('---------------------------------')
        
        # imag axis failed to converge
        if savedir is None: exit()

        wn, vn, ek, w, nB, nF, DRbareinv = self.setup_realaxis()

        if os.path.exists(savedir+'SR.npy'):
            if cont:
                print('CONTINUING FROM EXISTING REAL AXIS DATA')
                SR = np.load(savedir+'SR.npy')
                PIR = np.load(savedir+'PIR.npy')
            else:
                print('data exists. Not continuing. Set cont=True or delete data.')
                exit()
        
        else:
            SR  = np.zeros([self.nk,self.nk,self.nr], dtype=complex)
            PIR = np.zeros([self.nk,self.nk,self.nr], dtype=complex)

        GR  = self.compute_GR(w, ek, mu, SR)
        DR  = self.compute_DR(DRbareinv, PIR)
        
        # convert to imaginary frequency
        G = fourier.t2w(G, self.beta, self.dim, 'fermion')[0]

        # compute Gsum
        if self.renormalized:
            Gsum_plus  = np.zeros([self.nk,self.nk,self.nr], dtype=complex)
        Gsum_minus = np.zeros([self.nk,self.nk,self.nr], dtype=complex)
        for i in range(self.nr):
            if self.renormalized:
                Gsum_plus[:,:,i]  = np.sum(G/((w[i]+1j*wn)[None,None,:]), axis=2) / self.beta 
            Gsum_minus[:,:,i] = np.sum(G/((w[i]-1j*wn)[None,None,:]), axis=2) / self.beta
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
        frac = 0.8
        best_chg = None
        for i in range(100):
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

            if best_chg is None or np.mean(change) < best_chg:
                best_chg = np.mean(change)
                np.save('realchg.npy', [best_chg])
                np.save('savedir.npy', [savedir])            
                np.save(savedir+'w', w)
                np.save(savedir+'GR', GR)
                np.save(savedir+'SR', SR)
                np.save(savedir+'DR', DR)
                np.save(savedir+'PIR', PIR)
                
            

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
        
        #folder = 'data_unrenormalized_nk40_abstp0.000_dim2_g01.73691_nw256_omega0.100_dens0.000_beta40.0000_QNone/'
        #folder = 'data_renormalized_nk40_abstp0.000_dim2_g01.73691_nw256_omega0.100_dens0.000_beta20.0000_QNone/'
        
        params = read_params(basedir, folder)
        GR = np.load(os.path.join(basedir, 'data', folder, 'GR.npy'))
        
        nk = params['nk']
        plt.figure()
        plt.plot(-1/np.pi * GR[nk//2, nk//2, :].imag)
        plt.savefig(basedir+'data/'+folder+'G00')
        plt.close()
        
        A = -1/np.pi * GR[:, nk//2, :].imag
        plt.figure()
        plt.imshow(A.T, origin='lower', aspect='auto', extent=[-np.pi,np.pi,-0.2,0.2])
        plt.colorbar()
        plt.xlim(-np.pi/2, np.pi/2)
        plt.savefig(basedir+'data/'+folder+'Akw')
        plt.close()
        
        omegq = omega_q(params['omega'], J=0.020, nk=params['nk'])
        
        DR = np.load(os.path.join(basedir, 'data', folder, 'DR.npy'))
        
        B = np.zeros((nk, np.shape(DR)[2]))
        for ik in range(nk):
            B[ik] = -1/np.pi * DR[ik, nk//2, :].imag
        
        
        plt.figure()
        plt.imshow(B.T, origin='lower', aspect='auto', extent=[-np.pi,np.pi,-0.2,0.2])
        plt.colorbar()
        plt.xlim(-np.pi, np.pi)
        plt.ylim(0, 0.12)
        #plt.plot(np.arange(-np.pi, np.pi, 2*np.pi/nk), np.diag(omegq))
        plt.plot(np.arange(-np.pi, np.pi, 2*np.pi/nk), omegq[:, nk//2])
        plt.savefig(basedir+'data/'+folder+'Bkw')
        plt.close()
        
        
    plot()
    exit
    
    print('2D Renormalized Migdal Real Axis')

    params = {}
    params['nw']    = 256
    params['nk']    = 40
    params['t']     = 0.040
    params['tp']    = 0.0
    params['omega'] = 0.100
    params['omega_q'] = omega_q(params['omega'], J=0.020, nk=params['nk'])
    params['dens']  = 0.0
    params['renormalized'] = True
    params['sc']    = 0
    params['band']  = band_square_lattice
    params['beta']  = 20.0
    params['dim']   = 2
    
    params['fixed_mu'] = -0.090
    #params['fixed_mu'] = -0.00

    #basedir = '/home/groups/simes/bln/data/elph/imagaxis/example/'
    basedir = '../test/'
    if not os.path.exists(basedir): os.makedirs(basedir)

    #lamb = 1/6
    #lamb = 188.56
    W    = 8.0*params['t']
    #params['g0'] = mylamb2g0(lamb, params['omega'], W)
    params['g0'] = 1.73691219
    print('g0 is ', params['g0'])
    
    
    #params['q0'] = None
    params['q0'] = 0.3
    if params['q0'] is not None:
        params['gq2'] = gexp_2d(params['nk'], q0=params['q0'])**2
    

    params['dw']     = 0.001
    params['wmin']   = -1.0
    params['wmax']   = +1.0
    params['idelta'] = 0.005j
    
    #basedir = '/home/groups/simes/bln/data/elph/imagaxis/example/'
    basedir = '../test/'
    if not os.path.exists(basedir): os.makedirs(basedir)

    
    
    migdal = RealAxisMigdal(params, basedir)

    sc_iter, S0, PI0 = 100, None, None
    migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, mu0=None, frac=0.2, cont=False)

    

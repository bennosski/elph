from numpy import *
from convolution import conv
from base import MigdalBase
from functions import mylamb2g0, myg02lamb, band_square_lattice, gexp_2d, omega_q, read_params
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import fourier


class Migdal(MigdalBase):
    #------------------------------------------------------------
    def compute_fill(self, Gw):
        return real(1.0 + 2./(self.nk**2 * self.beta) * 2.0*np.sum(Gw))
    #------------------------------------------------------------
    def compute_n(self, Gtau):
        return -2.0*mean(Gtau[:,:,-1]).real


    def compute_n_tail(self, wn, ek, mu):
        baresum = 1 + 4 * np.mean(1/4 * np.tanh(-self.beta*(ek-mu)/2))
        bareGw = 1.0/(1j*wn[None,None,:] - (ek[:,:,None]-mu))
        return baresum - (1 + 2/(self.nk**2 * self.beta) * 2*np.sum(bareGw.real))


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
        elif hasattr(self, 'gk2'):
            return -self.g0**2/self.nk**2 * self.gk2[:,:,None] * conv(G, D, ['k-q,q','k-q,q'], [0,1], [True,True], self.beta)
        else:
            return -self.g0**2/self.nk**2 * conv(G, D, ['k-q,q','k-q,q'], [0,1], [True,True], self.beta)
    #------------------------------------------------------------
    def compute_GG(self, G):
        if hasattr(self, 'gq2'):
            return 2.0/self.nk**2 * self.gq2[:,:,None] * conv(G, -G[:,:,::-1], ['k,k+q','k,k+q'], [0,1], [True,True], self.beta)
        elif hasattr(self, 'gk2'):
            return 2.0/self.nk**2 * conv(self.gk2[:,:,None]*G, -G[:,:,::-1], ['k,k+q','k,k+q'], [0,1], [True,True], self.beta)
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
        if hasattr(self, 'gq2'):
            return 1 - self.g0**2 / self.nk**2 * conv(F0gamma, self.gq2[:,:,None]*D, ['k-q,q','k-q,q', 'm,n-m'], [0,1,2], [True,True,False], self.beta, kinds=('fermion','boson','fermion'), op='...,...', jumps=(jumpF0gamma, jumpD))
        elif hasattr(self, 'gk2'):
            return 1 - self.g0**2 * self.gk2[:,:,None] / self.nk**2 * conv(F0gamma, D, ['k-q,q','k-q,q', 'm,n-m'], [0,1,2], [True,True,False], self.beta, kinds=('fermion','boson','fermion'), op='...,...', jumps=(jumpF0gamma, jumpD))
        else:
            return 1 - self.g0**2/self.nk**2 * conv(F0gamma, D, ['k-q,q','k-q,q', 'm,n-m'], [0,1,2], [True,True,False], self.beta, kinds=('fermion','boson','fermion'), op='...,...', jumps=(jumpF0gamma, jumpD))


    def compute_jjcorr(self, G, D): 
        ''' 
        input is G(tau)
        returns jjcorr(q=0, tau)
        '''
        ks = np.arange(-np.pi, np.pi, 2*np.pi/self.nk)
        sin = np.sin(ks)
        cos = np.cos(ks)
        fk = (-2*self.t*sin[:,None] - 4*self.tp*sin[:,None]*cos[None,:])
        jj0t = 2.0 / self.nk**2 * np.sum((fk**2)[:,:,None] * G * G[:,:,::-1], axis=(0,1))

        np.save('jj0t{}'.format('r' if self.renormalized else 'u'), jj0t)

        Dw = fourier.t2w(D, self.beta, 2, 'boson')
         
        jj0v = fourier.t2w(jj0t, self.beta, 0, 'boson')        
        jjv = jj0v / (1 + self.g0**2 * Dw[self.nk//2, self.nk//2] * jj0v)

        ivn = 2*np.arange(len(jjv))*np.pi/self.beta

        np.save('jjv{}'.format('r' if self.renormalized else 'u'), jjv)
    


if __name__=='__main__':
    # example usage as follows :

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
        
        
    def run_forward_scattering():
        time0 = time.time()
    
        # example usage as follows :
        # forward-scattering parameters
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
        Xsc, Xcdw = migdal.susceptibilities(savedir, sc_iter, G, D, GG, frac=1.0)
        save(savedir + 'Xsc.npy',  [Xsc])
        save(savedir + 'Xcdw.npy', [Xcdw])
    
        print('------------------------------------------')
        print('simulation took', time.time()-time0, 's')
    
    def run_dwave(l, beta=16.0, S0=None, PI0=None):
        time0 = time.time()
    
        basedir = '../test_dwave/'
        if not os.path.exists(basedir): os.makedirs(basedir)
    
        print('2D Renormalized Migdal')
     
        params = {}
        params['nw']    = 128
        params['nk']    = 16
        params['t']     = 1
        params['tp']    = -0.3
        params['omega'] = 0.17
        params['dens']  = 0.8
        params['renormalized'] = True
        params['sc']    = False
        params['band']  = band_square_lattice
        params['beta']  = beta
        params['dim']   = 2
        params['fixed_mu'] = -1.11
        
        lamb = l
        W    = 8.0
        params['g0'] = mylamb2g0(lamb, params['omega'], W)
        print('g0 is ', params['g0'])
        
        ks = np.arange(-np.pi, np.pi, 2*np.pi/params['nk'])
        params['gk2'] = (np.cos(ks[:,None]) - np.cos(ks[None,:]))**2
        
        
        migdal = Migdal(params, basedir)
    
        sc_iter = 1000
        savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, frac=0.2)
        
        if G is None:
            return False, None, None
    
        PI = params['g0']**2 * GG
        save(savedir + 'S.npy', S)
        save(savedir + 'PI.npy', PI)
        save(savedir + 'G.npy', G)
        save(savedir + 'D.npy', D)
    
        sc_iter = 400
        Xsc, Xcdw = migdal.susceptibilities(savedir, sc_iter, G, D, GG, frac=1.0)
        save(savedir + 'Xsc.npy',  [Xsc])
        save(savedir + 'Xcdw.npy', [Xcdw])
    
        if Xcdw is None:
            return False, None, None
    
        print('------------------------------------------')
        print('simulation took', time.time()-time0, 's')
    
        return True, G, S
    
    def x_vs_T(l):
        
        beta = 1
        dbeta = 4
        S0, PI0 = None, None
        
        while beta < 80:
            
            res, A, B = run_dwave(l, beta, S0, PI0)
            
            if not res:
                break
            
            S0, PI0 = A, B
            beta += dbeta
        
    def plot_x_vs_t(l):
        basedir = '../test_dwave/'
        df = os.path.join(basedir, 'data/')
        folders = os.listdir(df)
        
        data = {}
        
        data['renormalized']   = {'betas':[], 'xscs': [], 'xcdws': []}
        data['unrenormalized'] = {'betas':[], 'xscs': [], 'xcdws': []}
        
        for folder in folders:
            params = read_params(basedir, folder)
            path = os.path.join(df, folder, 'Xsc.npy')
            pathcdw = os.path.join(df, folder, 'Xcdw.npy')
            
            
            lamb = myg02lamb(params['g0'], params['omega'], 8.0)
            #if params['nk']!=64: continue
            if not np.abs(lamb - l) < 1e-5:
                continue
    
            if not os.path.exists(path): continue
                    
            xsc = np.load(path, allow_pickle=True)[0]
            xcdw = np.load(pathcdw, allow_pickle=True)
            print('argmax', np.argmax(xcdw), 'shape xcdw', np.shape(xcdw))
            xcdw = np.amax(xcdw)
            r = 'renormalized' if params['renormalized'] else 'unrenormalized'
            
            if xsc is not None:
                data[r]['betas'].append(params['beta'])
                data[r]['xscs'].append(xsc)
                data[r]['xcdws'].append(xcdw)
    
        d = data['renormalized']
        rb, rx, rc = zip(*sorted(zip(d['betas'], d['xscs'], d['xcdws'])))
        d = data['unrenormalized']
        rb = np.array(rb)
        rx = np.array(rx)
        rc = np.array(rc)
        
        if len(d['betas'])>0:
            ub, ux, uc = zip(*sorted(zip(d['betas'], d['xscs'], d['xcdws'])))
            ub = np.array(ub)
            ux = np.array(ux)
            uc = np.array(uc)
        
        
        f, (ax1, ax2) = plt.subplots(1, 2)
        f.set_size_inches(6, 3)
        plt.title('lamb=1/6, n=0.8, 16x16, t\'=-0.3')
        
        '''
        ax1.plot(1/ub, 1/ux, 'k.-')
        ind = uc!=None
        ax1.plot(1/ub[ind], 20/uc[ind], 'k.--')
        ax1.legend(['1/Xsc', '20/Xcdw'])
        ax1.set_title('unrenormalized ME')
        ax1.set_xlabel('T', fontsize=13)
         
        ax1.set_xlim(0, 0.6)
        ax1.set_ylim(0, 6)
        '''
        
        ax2.plot(1/rb, 1/rx, 'k.-')
        ind = rc!=None
        ax2.plot(1/rb[ind], 20/rc[ind], 'k.--')
        
        #ax2.set_xlim(0, 0.25)
        #ax2.set_ylim(0, 6)
        ax2.legend(['1/Xsc', '20/Xcdw'])
        ax2.set_title('renormalized ME')
        ax2.set_xlabel('T', fontsize=13)
        
        
        plt.tight_layout()
     
        plt.savefig(basedir+'div_l')
        plt.close()
            
        #print(uc)
        #print('beta and x r', [(b,x) for b,x in zip(rb, rc)])


    l = 3/6    
    #x_vs_T(l)
    plot_x_vs_t(l)
    
    #run_dwave()
    #plot()
        

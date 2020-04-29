from numpy import *
import numpy as np
import time
from convolution import conv
import os
import sys
from scipy import optimize
from scipy.ndimage.filters import gaussian_filter
from functions import lamb2g0_ilya
import fourier
#import matplotlib 
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from anderson import AndersonMixing
from scipy.linalg import solve


class MigdalBase:
    #----------------------------------------------------------
    def __init__(self, params, basedir, loaddir=None):
        # basedir is the folder when results will be saved
        if not os.path.exists(basedir): os.makedirs(basedir)
        self.basedir = basedir
        self.keys = params.keys() 
        for key in params:
            setattr(self, key, params[key])
    #-----------------------------------------------------------
    def setup(self):
        print('\nParameters\n----------------------------')
        
        print('nw     = {}'.format(self.nw))
        print('nk     = {}'.format(self.nk))
        print('beta   = {:.3f}'.format(self.beta))
        print('omega  = {}'.format(self.omega))
        print('g0     = {:.3f}'.format(self.g0))
        print('tp     = {:.3f}'.format(self.tp))
        print('dens   = {}'.format(self.dens))
        print('renorm = {}'.format(self.renormalized))
        print('sc     = {}'.format(self.sc))
        print('Q      = {}'.format(self.Q))
        self.dim = len(shape(self.band(1, 1.0, self.tp)))        
        print('dim    = {}'.format(self.dim))
        
        savedir = self.basedir+'data/data_{}_nk{}_abstp{:.3f}_dim{}_g0{:.5f}_nw{}_omega{:.3f}_dens{:.3f}_beta{:.4f}_Q{}/'.format('renormalized' if self.renormalized else 'unrenormalized', self.nk, abs(self.tp), self.dim, self.g0, self.nw, self.omega, self.dens, self.beta, self.Q)
        if not os.path.exists(self.basedir+'data/'): os.mkdir(self.basedir+'data/')
        if not os.path.exists(savedir): os.mkdir(savedir)

        assert self.nk%2==0
        assert self.nw%2==0

        self.ntau = 2*self.nw + 1

        wn = (2*arange(self.nw)+1) * pi / self.beta
        vn = (2*arange(self.nw+1)) * pi / self.beta
        
        ek = self.band(self.nk, 1.0, self.tp, Q=self.Q)
        
        # estimate filling and dndmu at the desired filling
        mu = optimize.fsolve(lambda mu : 2.0*mean(1.0/(exp(self.beta*(ek-mu))+1.0))-self.dens, 0.0)[0]
        deriv = lambda mu : 2.0*mean(self.beta*exp(self.beta*(ek-mu))/(exp(self.beta*(ek-mu))+1.0)**2)
        dndmu = deriv(mu)
        
        print('mu optimized = %1.3f'%mu)
        print('dndmu = %1.3f'%dndmu)
        print('savedir = ',savedir)
        
        return savedir, wn, vn, ek, mu, dndmu
    #-----------------------------------------------------------
    def compute_fill(self, Gw): pass
    #-----------------------------------------------------------
    def compute_n(self, G): pass
    #-----------------------------------------------------------
    def compute_G(self, wn, ek, mu, S): pass
    #-----------------------------------------------------------
    def compute_D(self, vn, PI): pass
    #-----------------------------------------------------------
    def compute_S(self, G, D): pass
    #-----------------------------------------------------------
    def compute_GG(self, G): pass
    #-----------------------------------------------------------
    def init_selfenergies(self): pass
    #-----------------------------------------------------------


    def main_renormalized(self, frac, interp=None, cont=True):
        
        savedir, wn, vn, ek, mu, dndmu = self.setup()
        mu = -1.11
        
        St, PIt = None, None
        
        best_change = 1
        if interp:
            if len(os.listdir(savedir))>2:
                print('DATA ALREADY EXISTS. PLEASE DELETE FIRST')
                exit()

            # used for seeding a calculation with more k points from a calculation done with fewer k points
            S0 = interp.S
            PI0 = interp.PI 
        elif os.path.exists(savedir+'S.npy') and os.path.exists(savedir+'PI.npy'):
            print('\nImag-axis calculation already done!!!! \n USING EXISTING DATA!!!!!')
            St  = np.load(savedir+'S.npy')
            PIt = np.load(savedir+'PI.npy')
            mu0 = np.load(savedir+'mu.npy')[0]
            best_change = np.load(savedir+'bestchg.npy')[0]
            print('best_change', best_change)
            if not cont:
                print('NOT continuing with imag axis')
                mu, G = self.dyson_fermion(wn, ek, mu0, St, self.dim)
                D = self.dyson_boson(vn, PIt, self.dim)            
                return savedir, mu0, G, D, St, PIt / self.g0**2
            else:
                print('continuing with imag axis')

        for key in self.keys:
            np.save(savedir+key, [getattr(self, key)])
        
        
        # move to setup
        if self.sc:
            # superconducting case
            shp = ones(self.dim + 3, int)
            shp[-2] = 2
            shp[-1] = 2
            jumpG = -reshape(identity(2), shp)
        else:
            jumpG = -1
            
        
        if St is None and PIt is None:
            St, PIt = self.init_selfenergies()
        Sw, jumpS = fourier.t2w(St, beta=self.beta, axis=self.dim, kind='fermion')
        PIw = fourier.t2w(PIt, beta=self.beta, axis=self.dim, kind='boson')
        Dw  = self.compute_D(vn, PIw)
            
        change = [None, None]
        for it in range(200):
            Gw = self.compute_G(wn, ek, mu, Sw) 
            Dw  = self.compute_D(vn, PIw)
    
            Gw2 = Gw**2
            
            dJdS = 1 + self.g0**2 / (self.nk**2 * self.beta) * Dw[self.nk//2,self.nk//2,0] * Gw2
            
            '''
            G2w = np.concatenate((Gw, Gw), axis=0)
            G2w = np.concatenate((G2w, G2w), axis=1)
            G2w = G2w[::2, ::2]
            
            dJdPI = 1 - self.g0**2 / (self.nk**2 * self.beta) * (
                np.trace(np.einsum('...ab,bc->...ac', Gw2, Gw[0,0,0]), axis1=-1, axis2=-2) +
                np.trace(np.einsum('...ab,...bc->...ac', G2w, Gw2), axis1=-1, axis2=-2))
            '''
            
            Gt = fourier.w2t(Gw, self.beta, self.dim, 'fermion', jumpG)
            Dt = fourier.w2t(Dw, self.beta, self.dim, 'boson')
            
            St = self.compute_S(Gt, Dt)
            PIt = self.g0**2 * self.compute_GG(Gt)
            
            newSw, jumpS = fourier.t2w(St, beta=self.beta, axis=self.dim, kind='fermion')
            newPIw = fourier.t2w(PIt, beta=self.beta, axis=self.dim, kind='boson')
           
            change[0] = np.mean(np.abs(newSw-Sw))/np.mean(np.abs(newSw+Sw))
            change[1] = np.mean(np.abs(newPIw-PIw))/np.mean(np.abs(newPIw+PIw))
            odlro = np.mean(np.abs(Sw[...,0,1]))
            print('iter {} change = {:.5e} {:5e}  odlro = {:.5e}'.format(it, *change, odlro))
            
            chg = np.mean(change)        
            if chg < best_change:
                best_change = chg
                np.save(savedir+'bestchg.npy', [best_change, change[0], change[1]])
                np.save(savedir+'mu.npy', [mu])
                np.save(savedir+'S.npy', St)
                np.save(savedir+'PI.npy', PIt)
            
            Sw = Sw - frac /dJdS * (Sw - newSw)
            PIw = PIw - frac * (PIw - newPIw)
            
            if chg < 1e-14: break
            
        
    def main_unrenormalized(self):
        
        savedir, wn, vn, ek, mu, dndmu = self.setup()
        mu = -1.11
        
        # move to setup
        if self.sc:
            # superconducting case
            shp = ones(self.dim + 3, int)
            shp[-2] = 2
            shp[-1] = 2
            jumpG = -reshape(identity(2), shp)
        else:
            jumpG = -1
            
            
        St, PIt = self.init_selfenergies()
        Sw, jumpS = fourier.t2w(St, beta=self.beta, axis=self.dim, kind='fermion')
        PIw = fourier.t2w(PIt, beta=self.beta, axis=self.dim, kind='boson')
        Dw  = self.compute_D(vn, PIw)
        Dt = fourier.w2t(Dw, self.beta, self.dim, 'boson')
           
        tau3 = np.array([[1,0], [0,-1]])
        for it in range(200):
            Gw = self.compute_G(wn, ek, mu, Sw) 

            dJdS = 1 + self.g0**2 / (self.nk**2 * self.beta) * Dw[self.nk//2,self.nk//2,0] * \
                np.einsum('ab,...bc,cd->...ad', tau3, Gw**2, tau3)
           
            Gt = fourier.w2t(Gw, self.beta, self.dim, 'fermion', jumpG)
            
            St = self.compute_S(Gt, Dt)
           
            newSw, jumpS = fourier.t2w(St, beta=self.beta, axis=self.dim, kind='fermion')
      
            change = np.mean(np.abs(newSw-Sw))/np.mean(np.abs(newSw+Sw))
            odlro = np.mean(np.abs(Sw[...,0,1]))
            print('iter {} change = {:.5e}  odlro = {:.5e}'.format(it, change, odlro))
            
            Sw = Sw -  1/dJdS * (Sw - newSw)
       
            if change < 1e-14: break
          
            
        
        
        
    def main_renormalized_matrix(self, fracS, fracPI, interp=None, cont=True):
        # 37 iters to 1e-4 both for omega=1, beta=80, lamb=1/6
        
        savedir, wn, vn, ek, mu, dndmu = self.setup()
        mu = -1.11
        
        St, PIt = None, None
        
        best_change = 1
        if interp:
            if len(os.listdir(savedir))>2:
                print('DATA ALREADY EXISTS. PLEASE DELETE FIRST')
                exit()

            # used for seeding a calculation with more k points from a calculation done with fewer k points
            St = interp.S
            PIt = interp.PI 
        elif os.path.exists(savedir+'S.npy') and os.path.exists(savedir+'PI.npy'):
            print('\nImag-axis calculation already done!!!! \n USING EXISTING DATA!!!!!')
            St  = np.load(savedir+'S.npy')
            PIt = np.load(savedir+'PI.npy')
            mu0 = np.load(savedir+'mu.npy')[0]
            best_change = np.load(savedir+'bestchg.npy')[0]
            print('best_change', best_change)
            if not cont:
                print('NOT continuing with imag axis')
                mu, G = self.dyson_fermion(wn, ek, mu0, St, self.dim)
                D = self.dyson_boson(vn, PIt, self.dim)            
                return savedir, mu0, G, D, St, PIt / self.g0**2
            else:
                print('continuing with imag axis')

        for key in self.keys:
            np.save(savedir+key, [getattr(self, key)])
        
        
        
        # move to setup
        if self.sc:
            # superconducting case
            shp = ones(self.dim + 3, int)
            shp[-2] = 2
            shp[-1] = 2
            jumpG = -reshape(identity(2), shp)
        else:
            jumpG = -1
            
        if St is None and PIt is None:   
            St, PIt = self.init_selfenergies()
        Sw, jumpS = fourier.t2w(St, beta=self.beta, axis=self.dim, kind='fermion')
        PIw = fourier.t2w(PIt, beta=self.beta, axis=self.dim, kind='boson')
        Dw  = self.compute_D(vn, PIw)
        Dt = fourier.w2t(Dw, self.beta, self.dim, 'boson')
                
        change = [None, None]
        tau3 = np.array([[1,0], [0,-1]])
        for it in range(200):
            Gw = self.compute_G(wn, ek, mu, Sw) 
            Dw  = self.compute_D(vn, PIw)
            
            Gt = fourier.w2t(Gw, self.beta, self.dim, 'fermion', jumpG)
            Dt = fourier.w2t(Dw, self.beta, self.dim, 'boson')
            
            St = self.compute_S(Gt, Dt)
            PIt = self.g0**2 * self.compute_GG(Gt)
            
            newSw, jumpS = fourier.t2w(St, beta=self.beta, axis=self.dim, kind='fermion')
            newPIw = fourier.t2w(PIt, beta=self.beta, axis=self.dim, kind='boson')
           
            change[0] = np.mean(np.abs(newSw-Sw))/np.mean(np.abs(newSw+Sw))
            change[1] = np.mean(np.abs(newPIw-PIw))/np.mean(np.abs(newPIw+PIw))
            odlro = np.mean(np.abs(Sw[...,0,1]))
            print('iter {} change = {:.5e} {:5e}  odlro = {:.5e}'.format(it, *change, odlro))
            
            chg = np.mean(change)        
            if chg < best_change:
                best_change = chg
                np.save(savedir+'bestchg.npy', [best_change, change[0], change[1]])
                np.save(savedir+'mu.npy', [mu])
                np.save(savedir+'S.npy', St)
                np.save(savedir+'PI.npy', PIt)


            Dflat = np.reshape(Dw[:,:,0], [-1])
            Drmk = Dflat[:,None] - Dflat[None,:]
            
            tGw2t = np.einsum('ab,...bc,cd->...ad', tau3, Gw**2, tau3)
            tGw2t = np.reshape(tGw2t, [self.nk**2, self.nw, 2, 2])
            
            J = Sw - newSw
            J = np.reshape(J, [self.nk**2, self.nw, 2, 2])
            
            for n in range(self.nw):
                # dJdS will be nk^2 x nk^2
                
                #tG2t = np.einsum('ab,kbc,cd->kad', tau3, Gw2[:, n], tau3)
                
                for a in range(2):
                    for b in range(2):
                        dJdS = np.eye(self.nk**2) + self.g0**2 / (self.nk**2 * self.beta) * Drmk * tGw2t[None, :, n, a, b]
                
                        dS = solve(dJdS, J[:, n, a, b])
                        
                        Sw[:,:,n,a,b] -= fracS * np.reshape(dS, (self.nk, self.nk))
            
            PIw = (1-fracPI) * PIw + fracPI * newPIw
                    
            if np.mean(change) < 1e-14: break    
        
        
    def main_renormalized_matrix_full(self):
        # makes no sense
        # can't combine n=0 for fermion and m=0 for boson
        # would really need to combine all freqs and k points togeter
        # how about an average over frequency?????
        
        savedir, wn, vn, ek, mu, dndmu = self.setup()
        mu = -1.11
        
        # move to setup
        if self.sc:
            # superconducting case
            shp = ones(self.dim + 3, int)
            shp[-2] = 2
            shp[-1] = 2
            jumpG = -reshape(identity(2), shp)
        else:
            jumpG = -1
            
            
        St, PIt = self.init_selfenergies()
        Sw, jumpS = fourier.t2w(St, beta=self.beta, axis=self.dim, kind='fermion')
        PIw = fourier.t2w(PIt, beta=self.beta, axis=self.dim, kind='boson')
        Dw  = self.compute_D(vn, PIw)
        Dt = fourier.w2t(Dw, self.beta, self.dim, 'boson')
                
        change = [None, None]
        tau3 = np.array([[1,0], [0,-1]])
        for it in range(200):
            Gw = self.compute_G(wn, ek, mu, Sw) 
            Dw  = self.compute_D(vn, PIw)
            
            print('shapes ', np.shape(Gw), np.shape(Dw))
            exit()
            
            Gt = fourier.w2t(Gw, self.beta, self.dim, 'fermion', jumpG)
            Dt = fourier.w2t(Dw, self.beta, self.dim, 'boson')
            
            St = self.compute_S(Gt, Dt)
            PIt = self.g0**2 * self.compute_GG(Gt)
            
            newSw, jumpS = fourier.t2w(St, beta=self.beta, axis=self.dim, kind='fermion')
            newPIw = fourier.t2w(PIt, beta=self.beta, axis=self.dim, kind='boson')
           
            change[0] = np.mean(np.abs(newSw-Sw))/np.mean(np.abs(newSw+Sw))
            change[1] = np.mean(np.abs(newPIw-PIw))/np.mean(np.abs(newPIw+PIw))
            odlro = np.mean(np.abs(Sw[...,0,1]))
            print('iter {} change = {:.5e} {:5e}  odlro = {:.5e}'.format(it, *change, odlro))
            

            Dflat = np.reshape(Dw[:,:,0], [-1])
            Drmk = Dflat[:,None] - Dflat[None,:]
            
            Gflat = np.reshape(Gw[:,:,0], [-1,2,2])
            tG0kmrt = Gflat[None,:,:,:] - Gflat[:,None,:,:]
            tG0kmrt = np.einsum('ab,rkbc,cd->rkad', tau3, tG0kmrt, tau3)
            
            tGw2t = np.einsum('ab,...bc,cd->...ad', tau3, Gw**2, tau3)
            tGw2t = np.reshape(tGw2t, [self.nk**2, self.nw, 2, 2])
            
            Dw2 = Dw**2
            
            G2w = np.concatenate((Gw, Gw), axis=0)
            G2w = np.concatenate((G2w, G2w), axis=1)
            G2w = G2w[::2, ::2]
            G2w = np.reshape(G2w, [self.nk**2, self.nw, 2, 2])
            
            JS = Sw - newSw
            JS = np.reshape(JS, [self.nk**2, self.nw, 2, 2])
            JPI = PIw - newPIw
            JPI = np.reshape(JPI, [self.nk**2, self.nw, 2, 2])
            J = np.concatenate((JS, JPI), axis=0)
            
            for n in range(self.nw):
                # dJdS will be nk^2 x nk^2
                
                Gflat = np.reshape(Gw[:,:,n], [-1,2,2])
                Gkmr  = Gflat[None,:,:,:] - Gflat[:,None,:,:]
                
                #tGw2t = np.einsum('ab,kbc,cd->kad', tau3, Gw2[:, n], tau3)
                
                dJdS = np.zeros((2*self.nk**2, 2*self.nk**2))
                
                dJdS[self.nk**2:, :self.nk**2] = -self.g0**2 / (self.nk**2 * self.beta) * \
                            (np.einsum('kab, rkbc->rkac', tGw2t[:,n], Gkmr) + \
                             np.einsum('kab, kbc->kac', G2w[:,n], tGw2t[:,n])[None,:])
                                     
                dJdS[self.nk**2:, self.nk**2:] = np.eye(self.nk**2)
                
                for a in range(2):
                    for b in range(2):
                        dJdS[:self.nk**2, :self.nk**2] = np.eye(2*self.nk**2) + self.g0**2 / (self.nk**2 * self.beta) * Drmk * tG2t[None, :, n, a, b]
                
                        dJdS[:self.nk**2, self.nk**2:] = self.g0**2 / (self.nk**2 * self.beta) * Dw2[None,:,n] * tG0kmrt[:,:,a,b]
                        
                        dS = solve(dJdS, J[:, n, a, b])
                        
                        Sw[:,:,n,a,b] -= 0.9 * np.reshape(dS, (self.nk, self.nk))
            
            PIw = 0.1 * PIw + 0.9 * newPIw
                    
            if np.mean(change) < 1e-14: break    
        
        
        
    def main_unrenormalized_matrix(self):
        # ~ 22 iters to 1e-4 stable for omega=1, beta=80, lamb=0.3
        
        savedir, wn, vn, ek, mu, dndmu = self.setup()
        mu = -1.11
        
        # move to setup
        if self.sc:
            # superconducting case
            shp = ones(self.dim + 3, int)
            shp[-2] = 2
            shp[-1] = 2
            jumpG = -reshape(identity(2), shp)
        else:
            jumpG = -1
            
            
        St, PIt = self.init_selfenergies()
        Sw, jumpS = fourier.t2w(St, beta=self.beta, axis=self.dim, kind='fermion')
        PIw = fourier.t2w(PIt, beta=self.beta, axis=self.dim, kind='boson')
        Dw  = self.compute_D(vn, PIw)
        Dt = fourier.w2t(Dw, self.beta, self.dim, 'boson')
           
        Dflat = np.reshape(Dw[:,:,0], [-1])
        Drmk = Dflat[:,None] - Dflat[None,:]
                
        
        tau3 = np.array([[1,0], [0,-1]])
        for it in range(200):
            Gw = self.compute_G(wn, ek, mu, Sw) 

            Gt = fourier.w2t(Gw, self.beta, self.dim, 'fermion', jumpG)
            
            St = self.compute_S(Gt, Dt)
           
            newSw, jumpS = fourier.t2w(St, beta=self.beta, axis=self.dim, kind='fermion')
      
            change = np.mean(np.abs(newSw-Sw))/np.mean(np.abs(newSw+Sw))
            odlro = np.mean(np.abs(Sw[...,0,1]))
            print('iter {} change = {:.5e}  odlro = {:.5e}'.format(it, change, odlro))
            
            tGw2t = np.einsum('ab,...bc,cd->...ad', tau3, Gw**2, tau3)
            tGw2t = np.reshape(tGw2t, [self.nk**2, self.nw, 2, 2])
            
            J = Sw - newSw
            J = np.reshape(J, [self.nk**2, self.nw, 2, 2])

            J = Sw - newSw
            J = np.reshape(J, [self.nk**2, self.nw, 2, 2])
            
            for n in range(self.nw):
                # dJdS will be nk^2 x nk^2
                
                #tG2t = np.einsum('ab,kbc,cd->kad', tau3, Gw2[:, n], tau3)
                
                for a in range(2):
                    for b in range(2):
                        
                        dJdS = np.eye(self.nk**2) + self.g0**2 / (self.nk**2 * self.beta) * Drmk * tGw2t[None, :, n, a, b]
                
                        dS = solve(dJdS, J[:, n, a, b])
                        
                        Sw[:,:,n,a,b] -= 0.9 * np.reshape(dS, (self.nk, self.nk))
            
                    
            if change < 1e-14: break    
        
        
        
    def main_unrenormalized_wrong(self):
        # doesn't work because we need to minimize J^2 not find a root
        # i.e. using inverse of hessian
        
        
        savedir, wn, vn, ek, mu, dndmu = self.setup()
        mu = -1.11
        
        # move to setup
        if self.sc:
            # superconducting case
            shp = ones(self.dim + 3, int)
            shp[-2] = 2
            shp[-1] = 2
            jumpG = -reshape(identity(2), shp)
        else:
            jumpG = -1
            
            
        St, PIt = self.init_selfenergies()
        Sw, jumpS = fourier.t2w(St, beta=self.beta, axis=self.dim, kind='fermion')
        PIw = fourier.t2w(PIt, beta=self.beta, axis=self.dim, kind='boson')
        Dw  = self.compute_D(vn, PIw)
        Dt = fourier.w2t(Dw, self.beta, self.dim, 'boson')
        sumDw = np.sum(Dw)
        
        tau3 = np.array([[1,0], [0,-1]])
        for it in range(200):
            Gw = self.compute_G(wn, ek, mu, Sw) 

            #dJdS = 1 + self.g0**2 / (self.nk**2 * self.beta) * Dw[self.nk//2,self.nk//2,0] * \
            #    np.einsum('ab,...bc,cd->...ad', tau3, Gw**2, tau3)
           
            Gt = fourier.w2t(Gw, self.beta, self.dim, 'fermion', jumpG)
            
            St = self.compute_S(Gt, Dt)
           
            newSw, jumpS = fourier.t2w(St, beta=self.beta, axis=self.dim, kind='fermion')
      
        
            '''
            Jnk = Sw - newSw
            J = np.sum(Jnk**2)
            
            Jtk = fourier.w2t(Jnk, self.beta, self.dim, 'fermion', jump=0)
            convJDt = conv(Jtk, Dt[:,:,:,None,None], ['k-q,q','k-q,q'], [0,1], [True,True], beta=self.beta)
            convJDw, _ = fourier.t2w(convJDt, beta=self.beta, axis=self.dim, kind='fermion')
            
            print('shape Jnk', np.shape(Jnk))
            print('shape Gw', np.shape(Gw))
            print('convJDw', np.shape(convJDw))
            dJdS = 2*Jnk + 2*self.g0**2 / self.nk**2 * \
                np.einsum('ab,...bc,cd->...ad', tau3, Gw**2, tau3) * convJDw
            '''
            
            Jnk = Sw - newSw
            sJnk = np.sign(Jnk)
            J = np.sum(np.abs(Jnk))
            
            Jtk = fourier.w2t(sJnk, self.beta, self.dim, 'fermion', jump=0)
            convJDt = conv(Jtk, Dt[:,:,:,None,None], ['k-q,q','k-q,q'], [0,1], [True,True], beta=self.beta)
            convJDw, _ = fourier.t2w(convJDt, beta=self.beta, axis=self.dim, kind='fermion')
            
            print('shape Jnk', np.shape(Jnk))
            print('shape Gw', np.shape(Gw))
            print('convJDw', np.shape(convJDw))
            dJdS = sJnk + self.g0**2 / self.nk**2 * \
                np.einsum('ab,...bc,cd->...ad', tau3, Gw**2, tau3) * convJDw
           
            
            change = np.mean(np.abs(newSw-Sw))/np.mean(np.abs(newSw+Sw))
            odlro = np.mean(np.abs(Sw[...,0,1]))
            print('iter {} change = {:.5e}  odlro = {:.5e}'.format(it, change, odlro))
            
            Sw = Sw - 0.001 * J / dJdS
       
            if change < 1e-14: break
    



    def dyson_fermion(self, wn, ek, mu, S, axis):
        Sw, jumpS = fourier.t2w(S, self.beta, axis, 'fermion')

        mu = -1.11
        '''
        if abs(self.dens-1.0)>1e-10:
            mu_new = optimize.fsolve(lambda mu : self.compute_fill(self.compute_G(wn,ek,mu,Sw))-self.dens, mu)[0]
            mu = mu_new*0.9 + mu*0.1
        else:
            mu = 0
        '''

        Gw = self.compute_G(wn, ek, mu, Sw)        

        if len(shape(S))==self.dim + 3:
            # superconducting case
            shp = ones(self.dim + 3, int)
            shp[-2] = 2
            shp[-1] = 2
            jumpG = -reshape(identity(2), shp)
        else:
            jumpG = -1

        return mu, fourier.w2t(Gw, self.beta, axis, 'fermion', jumpG)
    #-----------------------------------------------------------
    def dyson_boson(self, vn, PI, axis):
        PIw = fourier.t2w(PI, self.beta, axis, 'boson')
        Dw  = self.compute_D(vn, PIw)
        return fourier.w2t(Dw, self.beta, axis, 'boson')


    def random_S(self):
        assert self.nk%2==0
        x = np.random.randn(self.nk+1, self.nk+1, self.ntau, 2, 2)
        x += x[::-1]
        x += x[:,::-1]
        x += np.transpose(x, axes=(1,0,2,3,4))
        x += np.transpose(x, axes=(0,1,2,4,3))
        return x[:-1, :-1] / 16
    

    def random_PI(self):
        assert self.nk%2==0
        x = np.random.randn(self.nk+1, self.nk+1, self.ntau)
        x += x[::-1]
        x += x[:,::-1]
        x += np.transpose(x, axes=(1,0,2))
        return x[:-1, :-1] / 8
    

    def gradient_descent(self, sc_iter, learning_rate, dr=1e-4, frac=0.9, alpha=None, S0=None, PI0=None, mu0=None, cont=False, interp=None):
        # stochastic descent selfconsistency
        
        savedir, wn, vn, ek, mu, dndmu = self.setup()

        np.save('savedir.npy', [savedir])

        best_change = 1
        if interp:
            if len(os.listdir(savedir))>2:
                print('DATA ALREADY EXISTS. PLEASE DELETE FIRST')
                exit()

            # used for seeding a calculation with more k points from a calculation done with fewer k points
            S0 = interp.S
            PI0 = interp.PI 
        elif os.path.exists(savedir+'S.npy') and os.path.exists(savedir+'PI.npy'):
            print('\nImag-axis calculation already done!!!! \n USING EXISTING DATA!!!!!')
            S0  = np.load(savedir+'S.npy')
            PI0 = np.load(savedir+'PI.npy')
            mu0 = np.load(savedir+'mu.npy')[0]
            best_change = np.load(savedir+'bestchg.npy')[0]
            print('best_change', best_change)
            if not cont:
                print('NOT continuing with imag axis')
                mu, G = self.dyson_fermion(wn, ek, mu0, S0, self.dim)
                D = self.dyson_boson(vn, PI0, self.dim)            
                return savedir, mu0, G, D, S0, PI0 / self.g0**2
            else:
                print('continuing with imag axis')


        for key in self.keys:
            np.save(savedir+key, [getattr(self, key)])
        

        print('\nImag-axis selfconsistency\n--------------------------')

        if S0 is None: 
            S0, PI0 = self.init_selfenergies()
        
        
        # store an S and a J
        # random update rup of S obeying symmetries
        # store S updated and S updated next -> Jnext
        # update S -= learning_rate * (Jnext - J) * rup
        # linear mixing between updated and old if desired (or more advanced optimizer)
        
        def step(S, PI):
            mu = None
            _, G = self.dyson_fermion(wn, ek, mu, S, self.dim)
            D = self.dyson_boson(vn, PI, self.dim)
            Snext  = self.compute_S(G, D)
            PInext = self.g0**2 * self.compute_GG(G)
            return Snext, PInext
        

        S0next, PI0next = step(S0, PI0)
        J0S  = np.mean(abs(S0next-S0))/np.mean(abs(S0next+S0))
        J0PI = np.mean(abs(PI0next-PI0))/np.mean(abs(PI0next+PI0))

        n = 0
        change = [0, 0]
        for i in range(sc_iter):
            rs  = self.random_S()
            rpi = self.random_PI()
            
            S1 = S0  + dr * rs 
            PI1 = PI0 + dr * rpi
            
            S1next, PI1next = step(S1, PI1)
            J1S  = np.mean(abs(S1next-S1))/np.mean(abs(S1next+S1))
            J1PI = np.mean(abs(PI1next-PI1))/np.mean(abs(PI1next+PI1))
            
            S1  =  S0 - learning_rate * (J1S - J0S) * rs
            PI1 = PI0 - learning_rate * (J1PI - J0PI) * rpi
            
            change[0] = np.mean(abs(S1-S0))/np.mean(abs(S1+S0))
            change[1] = np.mean(abs(PI1-PI0))/np.mean(abs(PI1+PI0))
    
            S0 = S1[:]
            PI0 = PI1[:]
            S0next, PI0next = step(S0, PI0)
            J0S  = np.mean(abs(S0next-S0))/np.mean(abs(S0next+S0))
            J0PI = np.mean(abs(PI0next-PI0))/np.mean(abs(PI0next+PI0))


            print('wtf', np.mean(np.abs(S1[...,0,1])))

            #if i%max(sc_iter//30,1)==0:
            if True:
                #odlro = ', ODLRO={:.4e}'.format(mean(abs(S[...,0,0,1]))) if self.sc else ''
                odlro = ', ODLRO={:.4e}'.format(np.amax(abs(S1[...,0,1]))) if self.sc else ''

                if len(np.shape(PI1)) == len(np.shape(S1)):
                    odrlo += ' {:.4e} '.format(np.amax(abs(PI1[...,0,1])))
                
                PImax = ', PImax={:.4e}'.format(np.amax(abs(PI1))) if self.renormalized else ''
                print('iter={} change={:.3e}, {:.3e} fill={:.13f} mu={:.5f}{}{}'.format(i, change[0], change[1], n, mu, odlro, PImax))

                #save(savedir+'S%d.npy'%i, S[self.nk//4,self.nk//4])
                #save(savedir+'PI%d.npy'%i, PI[self.nk//4,self.nk//4])

            #chg = 2*change[0]*change[1]/(change[0]+change[1]) 
            chg = np.mean(change)
            if chg < best_change:
                best_change = chg
                np.save(savedir+'bestchg.npy', [best_change, change[0], change[1]])
                np.save(savedir+'mu.npy', [mu])
                np.save(savedir+'S.npy', S1)
                np.save(savedir+'PI.npy', PI1)

            if i>10 and sum(change)<2e-14:
                # and abs(self.dens-n)<1e-5:
                break

        if sc_iter>1 and sum(change)>1e-5:
            # or abs(n-self.dens)>1e-3):
            print('Failed to converge')
            return None, None, None, None, None, None

        np.save(savedir+'G.npy', G)
        np.save(savedir+'D.npy', D)

        return savedir, mu, G, D, S1, PI1 / self.g0**2


    def zero_mixing_iter(self, x0, x1, f0, f1):
        e0 = x0 - f0
        e1 = x1 - f1
        me0 = np.mean(e0)
        me1 = np.mean(e1)
        #x0, x1 = x1, (e0*x1 - e1*x0) / (e0 - e1)
        frac = 0.99
        xn0 = x1
        #xn1 = frac*(e0*x1 - e1*x0) / (e0 - e1)  + (1-frac)*x1
        xn1 = frac*(me0*x1 - me1*x0) / (me0 - me1)  + (1-frac)*x1
        return xn0, xn1, e0, e1


    def selfconsistency1(self, sc_iter, frac=0.9, S0=None, PI0=None, mu0=None, cont=False, interp=None):
        savedir, wn, vn, ek, mu, dndmu = self.setup()

        np.save('savedir.npy', [savedir])

        best_change = 1
        if interp:
            if len(os.listdir(savedir))>2:
                print('DATA ALREADY EXISTS. PLEASE DELETE FIRST')
                exit()

            # used for seeding a calculation with more k points from a calculation done with fewer k points
            S0 = interp.S
            PI0 = interp.PI 
        elif os.path.exists(savedir+'S.npy') and os.path.exists(savedir+'PI.npy'):
            print('\nImag-axis calculation already done!!!! \n USING EXISTING DATA!!!!!')
            S0  = np.load(savedir+'S.npy')
            PI0 = np.load(savedir+'PI.npy')
            mu0 = np.load(savedir+'mu.npy')[0]
            best_change = np.load(savedir+'bestchg.npy')[0]
            print('best_change', best_change)
            if not cont:
                print('NOT continuing with imag axis')
                mu, G = self.dyson_fermion(wn, ek, mu0, S0, self.dim)
                D = self.dyson_boson(vn, PI0, self.dim)            
                return savedir, mu0, G, D, S0, PI0 / self.g0**2
            else:
                print('continuing with imag axis')


        for key in self.keys:
            np.save(savedir+key, [getattr(self, key)])
        
        if mu0 is not None:
            mu = mu0
        else:
            mu = mu

        print('\nImag-axis selfconsistency\n--------------------------')

        if S0 is None: 
            S0, PI0 = self.init_selfenergies()

        mu = -1.11
        def step(S, PI, mu):
            mu, G = self.dyson_fermion(wn, ek, mu, S, self.dim)
            D = self.dyson_boson(vn, PI, self.dim)
            S = self.compute_S(G, D)
            n = self.compute_n(G)

            if self.renormalized:
                GG = self.compute_GG(G)
                PI = self.g0**2 * GG
                
            return S, PI, GG, n
                           
        n = 0
        S1, PI1, GG1, n = step(S0, PI0, mu)
        fS0, fPI0 = S1.copy(), PI1.copy()

        frac = 0.4
        S1 = frac*S1 + (1-frac)*S0
        PI1 = frac*PI1 + (1-frac)*PI0

        fS1, fPI1, GG1, n = step(S1, PI1, mu)       
        GG = np.zeros_like(PI0)
        change = [0, 0]
        for i in range(sc_iter):
            fS0, fPI0 = fS1, fPI1
            fS1, fPI1, GG1, n = step(S1, PI1, mu)
            S0, S1, eS0, eS1 = self.zero_mixing_iter(S0, S1, fS0, fS1)
            PI0, PI1, ePI0, ePI1 = self.zero_mixing_iter(PI0, PI1, fPI0, fPI1)

            change = [np.mean(np.abs(eS1)), np.mean(np.abs(ePI1))]


            #if i%max(sc_iter//30,1)==0:
            if True:
                #odrlo = ', ODLRO={:.4e}'.format(mean(abs(S[...,0,0,1]))) if self.sc else ''
                odrlo = ', ODLRO={:.4e}'.format(np.amax(abs(S1[...,:,0,1]))) if self.sc else ''

                if len(np.shape(PI1)) == len(np.shape(S1)):
                    odrlo += ' {:.4e} '.format(np.amax(abs(PI1[...,:,0,1])))
                
                PImax = ', PImax={:.4e}'.format(np.amax(abs(PI1))) if self.renormalized else ''
                print('iter={} change={:.3e}, {:.3e} fill={:.13f} mu={:.5f}{}{}'.format(i, change[0], change[1], n, mu, odrlo, PImax))

                #save(savedir+'S%d.npy'%i, S[self.nk//4,self.nk//4])
                #save(savedir+'PI%d.npy'%i, PI[self.nk//4,self.nk//4])

            #chg = 2*change[0]*change[1]/(change[0]+change[1]) 
            chg = np.mean(change)
            if chg < best_change:
                best_change = chg
                np.save(savedir+'bestchg.npy', [best_change, change[0], change[1]])
                np.save(savedir+'mu.npy', [mu])
                np.save(savedir+'S.npy', S1)
                np.save(savedir+'PI.npy', PI1)

            if i>10 and sum(change)<2e-14:
                # and abs(self.dens-n)<1e-5:
                break

        if sc_iter>1 and sum(change)>1e-5:
            # or abs(n-self.dens)>1e-3):
            print('Failed to converge')
            return None, None, None, None, None, None

        np.save(savedir+'G.npy', G)
        np.save(savedir+'D.npy', D)

        '''
        if os.path.exists('savedirs.npy'):
            savedirs = list(np.load('savedirs.npy'))
        else:
            savedirs = []
        savedirs.append(savedir)
        np.save('savedirs.npy', savedirs)
        '''

        return savedir, mu, G, D, S1, GG1


    def random_frac(self):
        assert self.nk%2==0
        x = np.random.random(size=(self.nk+1, self.nk+1)) - 0.4
        x += x[::-1]
        x += x[:,::-1]
        x += x.T
        #x = gaussian_filter(x, self.nk*0.1, mode='wrap')
        return x[:-1, :-1] / 8
        '''
        x += x.T
        x += x[::-1].T
        x += x[:,::-1]
        x += x[::-1]
        return x[:-1, :-1] / 16
        '''


    def selfconsistency0(self, sc_iter, frac=0.9, alpha=None, S0=None, PI0=None, mu0=None, cont=False, interp=None):
        savedir, wn, vn, ek, mu, dndmu = self.setup()

        np.save('savedir.npy', [savedir])

        best_change = 1
        if interp:
            if len(os.listdir(savedir))>2:
                print('DATA ALREADY EXISTS. PLEASE DELETE FIRST')
                exit()

            # used for seeding a calculation with more k points from a calculation done with fewer k points
            S0 = interp.S
            PI0 = interp.PI 
        elif os.path.exists(savedir+'S.npy') and os.path.exists(savedir+'PI.npy'):
            print('\nImag-axis calculation already done!!!! \n USING EXISTING DATA!!!!!')
            S0  = np.load(savedir+'S.npy')
            PI0 = np.load(savedir+'PI.npy')
            mu0 = np.load(savedir+'mu.npy')[0]
            best_change = np.load(savedir+'bestchg.npy')[0]
            print('best_change', best_change)
            if not cont:
                print('NOT continuing with imag axis')
                mu, G = self.dyson_fermion(wn, ek, mu0, S0, self.dim)
                D = self.dyson_boson(vn, PI0, self.dim)            
                return savedir, mu0, G, D, S0, PI0 / self.g0**2
            else:
                print('continuing with imag axis')


        for key in self.keys:
            np.save(savedir+key, [getattr(self, key)])
        
        if mu0 is not None:
            mu = mu0
        else:
            mu = mu

        if alpha is not None:
            AMS  = AndersonMixing(alpha=alpha, frac=frac, n=2)
            AMPI = AndersonMixing(alpha=alpha, frac=frac, n=2)

        print('\nImag-axis selfconsistency\n--------------------------')

        if S0 is None: 
            S, PI = self.init_selfenergies()
        else:
            S, PI = S0[:], PI0[:]


        GG = np.zeros_like(PI)
        change = [0, 0]
        for i in range(sc_iter):
            S0, PI0  = S[:], PI[:]

            # compute G(tau), D(tau)
            mu, G = self.dyson_fermion(wn, ek, mu, S, self.dim)
            D = self.dyson_boson(vn, PI, self.dim)

            n = self.compute_n(G)
            #mu -= alpha*(n-self.dens)/dndmu
            #mu = optimize.fsolve(lambda mu : self.compute_n(self.compute_G(wn,ek,mu,S))-self.dens, mu)[0]

            '''
            if abs(self.dens-1.0)>1e-10:
                mu -= 0.01*(n-self.dens)/dndmu
            '''

            # compute new selfenergies S(tau) and PI(tau)
            S  = self.compute_S(G, D)
            change[0] = np.mean(abs(S-S0))/(np.mean(abs(S+S0))+1e-10)

            #gamma = 0.2
            #gamma = 0.2
            gamma = 1
            #r = self.random_frac()
            if alpha is None:
                #S  = gamma*frac*r[:,:,None,None,None]*S + (1-gamma*frac*r[:,:,None,None,None])*S0
                S  = gamma*frac*S + (1-gamma*frac)*S0
            else:
                S = AMS.step(S0, S)


            if self.renormalized:
                GG = self.compute_GG(G)
                PI = self.g0**2 * GG
                change[1] = np.mean(abs(PI-PI0))/(np.mean(abs(PI+PI0))+1e-10)
    
                #r = self.random_frac()
                if alpha is None:
                    #PI = r[:,:,None]*frac*PI + (1-r[:,:,None]*frac)*PI0
                    PI = frac*PI + (1-frac)*PI0
                else:
                    PI = AMPI.step(PI0, PI)

                # real part only?
                # PI = PI.real


            #if i%max(sc_iter//30,1)==0:
            if True:
                #odrlo = ', ODLRO={:.4e}'.format(mean(abs(S[...,0,0,1]))) if self.sc else ''
                odrlo = ', ODLRO={:.4e}'.format(np.amax(abs(S[...,:,0,1]))) if self.sc else ''

                if len(np.shape(PI)) == len(np.shape(S)):
                    odrlo += ' {:.4e} '.format(np.amax(abs(PI[...,:,0,1])))
                
                PImax = ', PImax={:.4e}'.format(np.amax(abs(PI))) if self.renormalized else ''
                print('iter={} change={:.3e}, {:.3e} fill={:.13f} mu={:.5f}{}{}'.format(i, change[0], change[1], n, mu, odrlo, PImax))

                #save(savedir+'S%d.npy'%i, S[self.nk//4,self.nk//4])
                #save(savedir+'PI%d.npy'%i, PI[self.nk//4,self.nk//4])

            #chg = 2*change[0]*change[1]/(change[0]+change[1]) 
            chg = np.mean(change)
            if chg < best_change:
                best_change = chg
                np.save(savedir+'bestchg.npy', [best_change, change[0], change[1]])
                np.save(savedir+'mu.npy', [mu])
                np.save(savedir+'S.npy', S)
                np.save(savedir+'PI.npy', PI)

            if i>10 and sum(change)<2e-14:
                # and abs(self.dens-n)<1e-5:
                break

        if sc_iter>1 and sum(change)>1e-5:
            # or abs(n-self.dens)>1e-3):
            print('Failed to converge')
            return None, None, None, None, None, None

        np.save(savedir+'G.npy', G)
        np.save(savedir+'D.npy', D)

        '''
        if os.path.exists('savedirs.npy'):
            savedirs = list(np.load('savedirs.npy'))
        else:
            savedirs = []
        savedirs.append(savedir)
        np.save('savedirs.npy', savedirs)
        '''

        return savedir, mu, G, D, S, GG
    #-------------------------------------------------------------------
    def susceptibilities(self, sc_iter, G, D, GG, frac=0.8): 
        print('\nComputing Susceptibilities\n--------------------------')

        assert self.sc==0

        # convert to imaginary frequency
        G  = fourier.t2w(G,  self.beta, self.dim, 'fermion')[0]
        D  = fourier.t2w(D,  self.beta, self.dim, 'boson')
        if self.renormalized:
            GG = fourier.t2w(GG, self.beta, self.dim, 'boson')

        F0 = G * conj(G)

        # confirm that F0 has no jump
        '''
        jumpF0 = np.zeros((self.nk, self.nk, 1))
        F0tau = fourier.w2t_fermion_alpha0(F0, self.beta, 2, jumpF0)

        figure()
        plot(F0tau[0,0].real)
        plot(F0tau[self.nk//2, self.nk//2].real)
        savefig(self.basedir+'F0tau')
        exit()
        '''

        #T  = ones([self.nk,self.nk,self.nw])

        jumpx = np.zeros([self.nk]*self.dim+[1])
        # momentum and frequency convolution 
        # the jump for F0 is zero
        jumpF0 = np.zeros([self.nk]*self.dim+[1])
        jumpD  = None
       
        x0 = self.compute_x0(F0, D, jumpF0, jumpD)

        jumpF0x = np.zeros([self.nk]*self.dim+[1], dtype=complex)

        iteration = 0
        x = np.zeros([self.nk]*self.dim+[self.nw], dtype=complex)
        while iteration < sc_iter:
            x_initial = x.copy()

            # compute jumpF0x
            F0x_tau = fourier.w2t(F0*x, self.beta, self.dim, 'fermion', jumpF0x)
            jumpF0x = F0x_tau[...,0] + F0x_tau[...,-1]
            jumpF0x = jumpF0x[...,None]

            x = x0 - self.compute_x0(F0*x, D, jumpF0x, jumpD)

            change = mean(abs(x - x_initial))/(mean(abs(x + x_initial))+1e-10)

            x = frac*x + (1-frac)*x_initial
            
            if change < 1e-10:
                break

            if iteration%max(sc_iter//20,1)==0:
                #print(f'change {change:.4e}')
                print('change ', change)
                
            iteration += 1

        #print(f'change {change:.4e}')
        print('change ', change)
        
        if change>1e-5:
            print('Susceptibility failed to converge')
            return None, None
    
        Xsc = 1.0 / (self.beta * self.nk**self.dim) * 2.0*sum(F0*(1+x)).real
        print(f'Xsc {Xsc:.4f}')

        # compute the CDW susceptibility
        Xcdw = None
        if self.renormalized:
            X0 = -GG[...,0]
            Xcdw = real(X0/(1.0 - 2.0*self.g0**2/self.omega * X0))

            Xcdw = ravel(Xcdw)
            a = argmax(abs(Xcdw))
            print('Xcdw = %1.4f'%Xcdw[a])

            if Xsc<0.0 or any(Xcdw<0.0): 
                print('Xcdw blew up')
                return None, None

        return Xsc, Xcdw


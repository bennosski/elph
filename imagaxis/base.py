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
import matplotlib 
from matplotlib.pyplot import *
from anderson import AndersonMixing


class MigdalBase:

    def __init__(self, params, basedir, loaddir=None):
        # basedir is the folder where results will be saved
        if not os.path.exists(basedir): os.makedirs(basedir)
        self.basedir = basedir
        self.keys = params.keys() 
        for key in params:
            setattr(self, key, params[key])


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
        if hasattr(self, 'sc'): print('sc     = {}'.format(self.sc))
        if hasattr(self, 'Q'): print('Q      = {}'.format(self.Q))
        self.dim = len(shape(self.band(1, 1.0, self.tp)))        
        if hasattr(self, 'dim'): print('dim    = {}'.format(self.dim))
        if hasattr(self, 'idelta'): print('idelta = {}'.format(self.idelta.imag))
        
        Q = None if not hasattr(self, 'Q') else self.Q

        savedir = self.basedir+'data/data_{}_nk{}_abstp{:.3f}_dim{}_g0{:.5f}_nw{}_omega{:.3f}_dens{:.3f}_beta{:.4f}_Q{}/'.format('renormalized' if self.renormalized else 'unrenormalized', self.nk, abs(self.tp), self.dim, self.g0, self.nw, self.omega, self.dens, self.beta, Q)
        if not os.path.exists(self.basedir+'data/'): os.mkdir(self.basedir+'data/')
        if not os.path.exists(savedir): os.mkdir(savedir)

        assert self.nk%2==0
        assert self.nw%2==0

        self.ntau = 2*self.nw + 1

        wn = (2*arange(self.nw)+1) * pi / self.beta
        vn = (2*arange(self.nw+1)) * pi / self.beta
        
        ek = self.band(self.nk, self.t, self.tp, Q)        
        
        '''        
        # estimate filling and dndmu at the desired filling
        mu = optimize.fsolve(lambda mu : 2.0*mean(1.0/(exp(self.beta*(ek-mu))+1.0))-self.dens, 0.0)[0]
        deriv = lambda mu : 2.0*mean(self.beta*exp(self.beta*(ek-mu))/(exp(self.beta*(ek-mu))+1.0)**2)
        dndmu = deriv(mu)
        '''

        mu = -1.11
        dndmu = None
 
        #print('mu optimized = %1.3f'%mu)
        #print('dndmu = %1.3f'%dndmu)       
        
        #mu = None
        #dndmu = None
        
        return savedir, wn, vn, ek, mu, dndmu


    def compute_fill(self, Gw): pass


    def compute_n(self, G): pass

   
    def compute_n_tail(self, wn, ek, mu): pass


    def compute_G(self, wn, ek, mu, S): pass


    def compute_D(self, vn, PI): pass


    def compute_S(self, G, D): pass


    def compute_GG(self, G): pass


    def init_selfenergies(self): pass


    def n_from_Sw(self, wn, ek, mu, Sw): pass


    def dyson_fermion(self, wn, ek, mu, S, axis):
        Sw, jumpS = fourier.t2w(S, self.beta, axis, 'fermion')

        if hasattr(self, 'fixed_mu'):
            mu = self.fixed_mu
        else:
            if abs(self.dens-1.0)>1e-10:
                mu_new = optimize.fsolve(lambda mu : self.compute_fill(self.compute_G(wn,ek,mu,Sw)) + self.compute_n_tail(wn, ek, mu) - self.dens, mu)[0]
                mu = mu_new*0.9 + mu*0.1
            else:
                mu = 0

        
        # old way : misses the contribution to n from the sum in the tails
        '''
        else:
            if abs(self.dens-1.0)>1e-10:
                mu_new = optimize.fsolve(lambda mu : self.compute_fill(self.compute_G(wn,ek,mu,Sw))-self.dens, mu)[0]
                mu = mu_new*0.9 + mu*0.1
            else:
                mu = 0
        '''


        '''
        # this is for density based on sum over iwn. 
        # But this is not jump corrected (missing the infinite tail so, probably it is less accurate than using the tau value)
        else:
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
   
    
    def dyson_boson(self, vn, PI, axis):
        PIw = fourier.t2w(PI, self.beta, axis, 'boson')
        Dw  = self.compute_D(vn, PIw)
        return fourier.w2t(Dw, self.beta, axis, 'boson')


    def selfconsistency(self, sc_iter, frac=0.9, alpha=None, S0=None, PI0=None, mu0=None, cont=False, interp=None):
        savedir, wn, vn, ek, mu, dndmu = self.setup()

        np.save('savedir.npy', [savedir])

        best_change = None

        if S0 is None and PI0 is None and os.path.exists(savedir+'S.npy') and os.path.exists(savedir+'PI.npy'):
            print('\nImag-axis calculation already done!!!! \n USING EXISTING DATA!!!!!')

            def load_with_backup(savedir, x):
                try:
                    out = np.load(savedir + x + '.npy')
                    print('regular load successful')
                except:
                    out = np.load(savedir + x + 'backup.npy')
                    print('backup load successful')
                return out

            #S0  = np.load(savedir+'S.npy')       
            #PI0 = np.load(savedir+'PI.npy')
            S0  = load_with_backup(savedir, 'S')
            PI0 = load_with_backup(savedir, 'PI')
            mu0 = np.load(savedir+'mu.npy')[0]
            best_change = np.mean(np.load(savedir+'bestchg.npy'))
            print('best_change', best_change)
            if not cont:
                print('NOT continuing with imag axis')
                mu, G = self.dyson_fermion(wn, ek, mu0, S0, self.dim)
                D = self.dyson_boson(vn, PI0, self.dim)            
                return savedir, mu0, G, D, S0, PI0 / self.g0**2
            else:
                print('continuing with imag axis')
        elif interp:
            if len(os.listdir(savedir))>2:
                print('DATA ALREADY EXISTS. PLEASE DELETE FIRST')
                exit()

            # used for seeding a calculation with more k points from a calculation done with fewer k points
            S0 = interp.S
            PI0 = interp.PI 
        

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

        '''
        # understand filling differences
        mu, G = self.dyson_fermion(wn, ek, mu, S0, self.dim)
        n_tau = self.compute_n(G)
        Gw, jump = fourier.t2w(G, self.beta, 2, 'fermion')
        print('Gw shape', Gw.shape)
        n_w = self.compute_fill(Gw)
        print('n_tau', n_tau)
        print('n_w', n_w)
        print('expected missing part')
        mp = self.compute_n_tail(wn, ek, mu)
        print('mp', mp)
        print('correct fill', n_w + mp)
        exit()
        '''
        

        if S0 is None: 
            S, PI = self.init_selfenergies()
        else:
            S, PI = S0[:], PI0[:]

        GG = np.zeros_like(PI)
        #GG = None
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
                mu -= 0.1*(n-self.dens)/dndmu
            '''

            # compute new selfenergies S(tau) and PI(tau)
            S  = self.compute_S(G, D)
            change[0] = np.mean(abs(S-S0))/(np.mean(abs(S+S0))+1e-10)


            if alpha is None:
                S = frac*S + (1-frac)*S0
            else:
                S = AMS.step(S0, S)


            if self.renormalized:
                GG = self.compute_GG(G)
                PI = self.g0**2 * GG
                change[1] = np.mean(abs(PI-PI0))/(np.mean(abs(PI+PI0))+1e-10)
    
                if alpha is None:
                    PI = frac*PI + (1-frac)*PI0
                else:
                    PI = AMPI.step(PI0, PI)


            if True:
                hasodrlo = True if ((hasattr(self, 'sc') and self.sc) or (hasattr(self, 'cdw') and self.cdw)) else False
                odrlo = ', ODLRO={:.4e}'.format(np.amax(abs(S[...,:,0,1]))) if hasodrlo else ''

                if hasodrlo and len(np.shape(PI)) == len(np.shape(S)):
                    odrlo += ' {:.4e} '.format(np.amax(abs(PI[...,:,0,1])))
                
                PImax = ', PImax={:.4e}'.format(np.amax(abs(PI))) if self.renormalized else ''
                print('iter={} change={:.3e}, {:.3e} fill={:.13f} mu={:.5f}{}{}'.format(i, change[0], change[1], n, mu, odrlo, PImax))


            chg = np.mean(change)
            if best_change is None or chg < best_change:
                best_change = chg
                np.save(savedir+'bestchg.npy', change)
                np.save(savedir+'mu.npy', [mu])
                np.save(savedir+'S.npy', S)
                np.save(savedir+'PI.npy', PI)

                np.save(savedir+'mubackup.npy', mu)
                np.save(savedir+'Sbackup.npy', S)
                np.save(savedir+'PIbackup.npy', PI)
      


            if sum(change)<2e-14:
                # and abs(self.dens-n)<1e-5:
                break

        if (sc_iter>1 and sum(change)>1e-5) or np.abs(self.dens-n)>1:
            # or abs(n-self.dens)>1e-3):
            print('Failed to converge')
            return None, None, None, None, None, None

        np.save(savedir+'G.npy', G)
        np.save(savedir+'D.npy', D)

        return savedir, mu, G, D, S, GG


    def susceptibilities(self, savedir, sc_iter, G, D, GG, frac=0.8): 
        print('\nComputing Susceptibilities\n--------------------------')

        assert self.sc==0

        if not self.renormalized:
            GG = self.compute_GG(G)

        # convert to imaginary frequency
        G  = fourier.t2w(G,  self.beta, self.dim, 'fermion')[0]
        D  = fourier.t2w(D,  self.beta, self.dim, 'boson')
        
        GG = fourier.t2w(GG, self.beta, self.dim, 'boson')



            
        F0 = G * conj(G)

        # confirm that F0 has no jump
        '''
        jumpF0 = np.zeros((self.nk, self.nk, 1))
        F0tau = fourier.w2t_fermion_alpha0(F0, self.beta, 2, jumpF0)

        figure()
        plot(F0tau[0,0].real)
        plot(F0tau[self.nk//2, self.nk//2].rel)
        savefig(self.basedir+'F0tau')
        exit()
        '''

        jumpx = np.zeros([self.nk]*self.dim+[1])
        # momentum and frequency convolution 
        # the jump for F0 is zero
        jumpF0 = np.zeros([self.nk]*self.dim+[1])
        jumpD  = None
       
        #gamma0 = 1 #self.compute_gamma(F0, D, jumpF0, jumpD)
        path = os.path.join(savedir, 'gamma.npy')
        if os.path.exists(path):
            gamma = np.load(path)
        else:
            gamma = np.ones([self.nk]*self.dim+[self.nw], dtype=complex)

        jumpF0gamma = np.zeros([self.nk]*self.dim+[1], dtype=complex)

        iteration = 0
        #gamma = np.ones([self.nk]*self.dim+[self.nw], dtype=complex)
        while iteration < sc_iter:
            gamma0 = gamma.copy()
            
            F0gamma = F0*gamma
            
            # compute jumpF0gamma
            F0gamma_tau = fourier.w2t(F0gamma, self.beta, self.dim, 'fermion', jumpF0gamma)
            jumpF0gamma = F0gamma_tau[...,0] + F0gamma_tau[...,-1]
            jumpF0gamma = jumpF0gamma[...,None]
            
            #print('size of jumpF0gamma {:.5e}'.format(np.amax(np.abs(jumpF0gamma))))

            gamma = self.compute_gamma(F0gamma, D, jumpF0gamma, jumpD)

            change = mean(abs(gamma - gamma0))/(mean(abs(gamma + gamma0))+1e-10)

            gamma = frac*gamma + (1-frac)*gamma0
            
            if change < 1e-10:
                break

            #if iteration%max(sc_iter//20,1)==0:
                #print(f'change {change:.4e}')
            print('change ', change)
                
            iteration += 1

        #print(f'change {change:.4e}')
        print('change ', change)
        
        if change>1e-5:
            print('Susceptibility failed to converge')
            return None, None

        np.save(savedir+'gamma.npy', gamma)
    
        #Xsc = 1.0 / (self.beta * self.nk**self.dim) * 2.0*sum(F0*(1+x)).real
        Xsc = 1.0 / (self.beta * self.nk**self.dim) * 2.0*sum(F0*gamma).real
        print(f'Xsc {Xsc:.4f}')


        # compute the CDW susceptibility
        Xcdw = None
        
        X0 = -GG[...,0] 
        Xcdw = real(X0/(1.0 - 2.0*self.g0**2/self.omega * X0))

        Xcdw = ravel(Xcdw)
        a = argmax(abs(Xcdw))
        print('Xcdw = %1.4f'%Xcdw[a])

        if any(Xcdw<0.0) or np.isnan(a): 
            print('Xcdw blew up')
            return Xsc, None


        return Xsc, Xcdw


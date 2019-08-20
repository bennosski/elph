from numpy import *
import time
from convolution import conv
import os
import sys
from scipy import optimize
from functions import lamb2g0_ilya
import fourier
import matplotlib 
matplotlib.use('TkAgg')
from matplotlib.pyplot import *
from anderson import AndersonMixing

class MigdalBase:
    #----------------------------------------------------------
    def __init__(self, params, basedir):
        # basedir is the folder where results will be saved
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
        print('SC     = {}'.format(self.sc))
        self.dim = len(shape(self.band(1, 1.0, self.tp)))        
        print('dim    = {}'.format(self.dim))
        
        savedir = self.basedir+'data/data_{}_nk{}_abstp{:.3f}_dim{}_g0{:.5f}_nw{}_omega{:.3f}_dens{:.3f}_beta{:.4f}/'.format('renormalized' if self.renormalized else 'unrenormalized', self.nk, abs(self.tp), self.dim, self.g0, self.nw, self.omega, self.dens, self.beta)
        if not os.path.exists(self.basedir+'data/'): os.mkdir(self.basedir+'data/')
        if not os.path.exists(savedir): os.mkdir(savedir)

        assert self.nk%2==0
        assert self.nw%2==0

        self.ntau = 2*self.nw + 1

        wn = (2*arange(self.nw)+1) * pi / self.beta
        vn = (2*arange(self.nw+1)) * pi / self.beta
        
        ek = self.band(self.nk, 1.0, self.tp)

        # estimate filling and dndmu at the desired filling
        mu = optimize.fsolve(lambda mu : 2.0*mean(1.0/(exp(self.beta*(ek-mu))+1.0))-self.dens, 0.0)[0]
        deriv = lambda mu : 2.0*mean(self.beta*exp(self.beta*(ek-mu))/(exp(self.beta*(ek-mu))+1.0)**2)
        dndmu = deriv(mu)

        print('mu optimized = %1.3f'%mu)
        print('dndmu = %1.3f'%dndmu)
        
        return savedir, wn, vn, ek, mu, deriv, dndmu
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
    def dyson_fermion(self, wn, ek, mu, S, axis):
        Sw, jumpS = fourier.t2w(S, self.beta, axis, 'fermion')
        
        mu = optimize.fsolve(lambda mu : self.compute_fill(self.compute_G(wn,ek,mu,Sw))-self.dens, mu)[0]

        Gw = self.compute_G(wn, ek, mu, Sw)        
        jumpG = -1 #np.ones((self.nk, self.nk, 1))
        return mu, fourier.w2t(Gw, self.beta, axis, 'fermion', jumpG)
    #-----------------------------------------------------------
    def dyson_boson(self, vn, PI, axis):
        PIw = fourier.t2w(PI, self.beta, axis, 'boson')
        Dw  = self.compute_D(vn, PIw)
        return fourier.w2t(Dw, self.beta, axis, 'boson')
    #-----------------------------------------------------------
    def selfconsistency(self, sc_iter, nmix=3, frac=0.9, alpha=0.5, S0=None, PI0=None, mu0=None):
        savedir, wn, vn, ek, mu, deriv, dndmu = self.setup()
        
        if mu0 is not None:
            mu = mu0

        AMS  = AndersonMixing(alpha=0.8)
        AMPI = AndersonMixing(alpha=0.8)
        #AMmu = AndersonMixing(alpha=0.1)

        print('\nSelfconsistency\n--------------------------')

        if S0 is None or PI0 is None: 
            S, PI = self.init_selfenergies()
        else:
            S, PI  = S0, PI0

        change = [0, 0]
        for i in range(sc_iter):
            S0, PI0  = S[:], PI[:]

            # compute G(tau), D(tau)
            mu, G = self.dyson_fermion(wn, ek, mu, S, self.dim)
            D = self.dyson_boson(vn, PI, self.dim)

            n = self.compute_n(G)
            #mu -= alpha*(n-self.dens)/dndmu
            #mu = optimize.fsolve(lambda mu : self.compute_n(self.compute_G(wn,ek,mu,S))-self.dens, mu)[0]

            # compute new selfenergies S(tau) and PI(tau)
            S  = self.compute_S(G, D)
            change[0] = mean(abs(S-S0))/(mean(abs(S+S0))+1e-10)
            S  = frac*S + (1-frac)*S0
            #S = AMS.step(S0, S)

            GG = self.compute_GG(G)
            PI = self.g0**2 * GG
            change[1] = mean(abs(PI-PI0))/(mean(abs(PI+PI0))+1e-10)
            PI = frac*PI + (1-frac)*PI0
            #PI = AMPI.step(PI0, PI)

            #if i%max(sc_iter//30,1)==0:
            if True:
                print('iter={} change={:.3e}, {:.3e} fill={:.13f} mu={:.5f}'.format(i, change[0], change[1], n, mu))

            if i>10 and change[0]<1e-14 and change[1]<1e-14:
                # and abs(self.dens-n)<1e-5:
                break

        if sc_iter>1 and (change[0]>1e-5 or change[1]>1e-5):
            # or abs(n-self.dens)>1e-3):
            print('Failed to converge')
            return None, None, None, None, None

        for key in self.keys:
            save(savedir+key, [getattr(self, key)])

        return savedir, mu, G, D, S, GG
    #---------------------------------------------------------------------------
    def susceptibilities(self, sc_iter, G, D, GG, frac=0.8): 
        print('\nComputing Susceptibilities\n--------------------------')

        assert self.sc==0

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
                print(f'change {change:.4e}')

            iteration += 1

        print(f'change {change:.4e}')

        if change>1e-5:
            print('Susceptibility failed to converge')
            return None, None
    
        Xsc = 1.0 / (self.beta * self.nk**self.dim) * 2.0*sum(F0*(1+x)).real
        print(f'Xsc {Xsc:.4f}')

        # compute the CDW susceptibility
        X0 = -GG[...,0]
        Xcdw = real(X0/(1.0 - 2.0*self.g0**2/self.omega * X0))

        Xcdw = ravel(Xcdw)
        a = argmax(abs(Xcdw))
        print('Xcdw = %1.4f'%Xcdw[a])

        if Xsc<0.0 or any(Xcdw<0.0): 
            print('Xcdw blew up')
            return None, None

        return Xsc, Xcdw


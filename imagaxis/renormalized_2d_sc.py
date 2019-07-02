from numpy import *
import time
from convolution import conv
import os
import sys
from scipy import optimize
from params import params, lamb2g0
import fourier

import matplotlib
matplotlib.use('agg')
from matplotlib.pyplot import *


class Migdal:
    tau0 = array([[1.0, 0.0], [0.0, 1.0]])
    tau1 = array([[0.0, 1.0], [1.0, 0.0]])
    tau3 = array([[1.0, 0.0], [0.0,-1.0]])

    #---------------------------------------------------------------------------
    def __init__(self, params, basedir):
        self.basedir = basedir
        for key in params:
            setattr(self, key, params[key])
    #---------------------------------------------------------------------------
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
        print('dim    = {}'.format(len(shape(self.band(1, params['t'], params['tp'])))))     

        savedir = self.basedir+'data/data_{}_nk{}_abstp{:.3f}_dim{}_g0{:.5f}_nw{}_omega{:.3f}_dens{:.3f}_sc{}/'.format('renormalized' if self.renormalized else 'unrenormalized', self.nk, abs(self.tp), len(shape(self.band(1, params['t'], params['tp']))), self.g0, self.nw, self.omega, self.dens, self.beta, self.sc)
        if not os.path.exists(self.basedir+'data/'): os.mkdir(self.basedir+'data/')
        if not os.path.exists(savedir): os.mkdir(savedir)

        assert self.nk%2==0
        assert self.nw%2==0

        self.ntau = 2*self.nw + 1

        wn = (2*arange(self.nw)+1) * pi / self.beta
        vn = (2*arange(self.nw+1)) * pi / self.beta
        
        ek = self.band(self.nk, params['t'], params['tp'])

        # estimate filling and dndmu at the desired filling
        mu = optimize.fsolve(lambda mu : 2.0*mean(1.0/(exp(self.beta*(ek-mu))+1.0))-self.dens, 0.0)[0]
        deriv = lambda mu : 2.0*mean(self.beta*exp(self.beta*(ek-mu))/(exp(self.beta*(ek-mu))+1.0)**2)
        dndmu = deriv(mu)

        print('mu optimized = %1.3f'%mu)
        print('dndmu = %1.3f'%dndmu)
        print('band bottom = %1.3f'%(ek[self.nk//2, self.nk//2]-mu))
        print('band = %1.3f'%ek[self.nk//2,self.nk//2])

        return savedir, wn, vn, ek, mu, deriv, dndmu
    #---------------------------------------------------------------------------
    def compute_n(self, G):
        # check the -1 here......
        #return -2.0*mean(G[:,:,-1,0,0]).real
        #return 2.0 + mean(G[:,:,0,0,0].real - G[:,:,0,1,1].real)
        nup = -mean(G[:,:,-1,0,0].real)
        ndw = -mean(G[:,:,0,1,1].real)        
        return nup + ndw
    #---------------------------------------------------------------------------
    def compute_G(self, wn, ek, mu, S):
        #return linalg.inv(1j*wn[None,None,:,None,None]*Migdal.tau0[None,None,None,:,:] - (ek[:,:,None,None,None]-mu)*Migdal.tau3[None,None,None,:,:] - S)

        #print('S', mean(abs(S)))
        
        #G = linalg.inv(1j*wn[None,None,:,None,None]*Migdal.tau0[None,None,None,:,:] - (ek[:,:,None,None,None]-mu)*Migdal.tau3[None,None,None,:,:] - S)
        #print('G off diag')
        #print(1.0/(self.beta*self.nk**2)*sum(G[:,:,:,0,1]).real)

        return linalg.inv(1j*wn[None,None,:,None,None]*Migdal.tau0[None,None,None,:,:] - (ek[:,:,None,None,None]-mu)*Migdal.tau3[None,None,None,:,:] - S)
    #---------------------------------------------------------------------------
    def compute_D(self, vn, PI):
        return 1.0/(-((vn**2)[None,None,:] + self.omega**2)/(2.0*self.omega) - PI)
    #---------------------------------------------------------------------------
    def compute_S(self, G, D):
        tau3Gtau3 = einsum('ab,xywbc,cd->xywad', Migdal.tau3, G, Migdal.tau3)
        return -self.g0**2/self.nk**2 * conv(tau3Gtau3, D[:,:,:,None,None], ['k-q,q','k-q,q'], [0,1], [True,True], self.beta)
    #---------------------------------------------------------------------------
    def compute_PI(self, G):
        tau3G = einsum('ab,...bc->...ac', Migdal.tau3, G)
        #                             CHECK THE 0.5 HERE!
        return 2.0*self.g0**2/self.nk**2 * 0.5*einsum('...aa->...', conv(tau3G, -tau3G[:,:,::-1,:,:], ['k,k+q','k,k+q'], [0,1], [True,True], self.beta, op='...ab,...bc->...ac'))
    #---------------------------------------------------------------------------
    def dyson_fermion(self, wn, ek, mu, S, axis):
        Sw, jumpS = fourier.t2w_fermion_alpha0(S, self.beta, axis)
        Gw = self.compute_G(wn, ek, mu, Sw)
        jumpG = -Migdal.tau0[None,None,None,:,:]
        return fourier.w2t_fermion_alpha0(Gw, self.beta, axis, jumpG)

    def dyson_boson(self, vn, PI, axis):
        PIw = fourier.t2w_boson(PI, self.beta, axis)
        Dw  = self.compute_D(vn, PIw)
        return fourier.w2t_boson(Dw, self.beta, axis)

    #---------------------------------------------------------------------------
    def selfconsistency(self, sc_iter, frac=0.9, alpha=0.5, S0=None, PI0=None):
        savedir, wn, vn, ek, mu, deriv, dndmu = self.setup()

        #print('wn')
        #print(wn)
        #exit()
        
        print('\nSelfconsistency\n--------------------------')

        if S0 is None or PI0 is None:
            #S  = self.sc*0.01*ones([self.nk,self.nk,self.ntau,2,2], dtype=complex)*Migdal.tau1[None,None,None,:,:]
            S  = zeros([self.nk,self.nk,self.ntau,2,2], dtype=complex)
            S[:,:,0,0,1] = 0.01 * self.sc
            S[:,:,0,1,0] = 0.01 * self.sc

            #S  = self.sc*0.01*ones([self.nk,self.nk,self.ntau,2,2], dtype=complex)*Migdal.tau1[None,None,None,:,:]
            PI = zeros([self.nk,self.nk,self.ntau], dtype=complex)
        else:
            S  = S0
            PI = PI0


        '''
        G = self.dyson_fermion(wn, ek, mu, S, 2)
        print('offdiagonal ', mean(G[:,:,-1,0,1]).real)
        figure()
        plot(G[self.nk//2, 0, :, 0, 0].real)
        plot(G[self.nk//2, 0, :, 0, 1].real)
        plot(G[self.nk//2, 0, :, 0, 1].imag)
        savefig('Gtau')
        exit()
        '''


        change = [0, 0]
        for i in range(sc_iter):
            S0  = S[:]
            PI0 = PI[:]

            # compute Gtau, Dtau

            G = self.dyson_fermion(wn, ek, mu, S, 2)
            D = self.dyson_boson(vn, PI, 2)

            #plot(1.0/self.nk**2 * sum(G[:,:,:,0,1], axis=(0,1)).real)
            #plot(1.0/self.nk**2 * sum(G[:,:,:,0,0], axis=(0,1)).real)

            n = self.compute_n(G)
            mu -= alpha*(n-self.dens)/dndmu
            #mu = optimize.fsolve(lambda mu : self.compute_fill(self.compute_G(wn,ek,mu,S))-self.dens, mu)[0]

            # compute new selfenergy

            S  = self.compute_S(G, D)
            change[0] = mean(abs(S-S0))/mean(abs(S+S0))
            S  = frac*S + (1-frac)*S0

            PI = self.compute_PI(G)
            change[1] = mean(abs(PI-PI0))/mean(abs(PI+PI0))
            PI = frac*PI + (1-frac)*PI0

            if i%1==0:
                print('change={:.3e}, {:.3e} fill={:.5f} mu={:.5f} ODLRO={:3e}'.format(change[0], change[1], n, mu, mean(G[:,:,-1,0,1]).real))

            if params['g0']<1e-10: break

            if i>10 and change[0]<1e-14 and change[1]<1e-14: break

        figure()
        plot(1.0/self.nk**2 * sum(G[:,:,:,0,1], axis=(0,1)).real)
        savefig('Gtau')


        if change[0]>1e-5 or change[1]>1e-5 or abs(n-self.dens)>1e-3:
            print('Failed to converge')
            return None, None, None, None, None

        save(savedir+'nw',    [self.nw])
        save(savedir+'nk',    [self.nk])
        save(savedir+'t',     [self.t])
        save(savedir+'tp',    [self.tp])
        save(savedir+'beta',  [self.beta])
        save(savedir+'omega', [self.omega])
        save(savedir+'g0',    [self.g0])
        save(savedir+'dens',  [self.dens])
        save(savedir+'renormalized', [self.renormalized])
        save(savedir+'sc',    [self.sc])
        
        return savedir, G, D, S, PI

if __name__=='__main__':

    # example usage as follows :

    print('2D Renormalized Migdal')

    params['beta'] = 100.0
    params['omega'] = 2.0
    lamb = 0.4
    W    = 8.0
    params['g0'] = lamb2g0(lamb, params['omega'], W)
    print('g0 is ', params['g0'])
    
    migdal = Migdal(params)

    sc_iter = 300
    S0, PI0  = None, None
    savedir, G, D, S, PI = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, frac=0.2)
    #save(savedir + 'S.npy', S)
    #save(savedir + 'PI.npy', PI)
    #save(savedir + 'G.npy', G)
    #save(savedir + 'D.npy', D)


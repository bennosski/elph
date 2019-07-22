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

        savedir = self.basedir+'data/data_{}_nk{}_abstp{:.3f}_dim{}_g0{:.5f}_nw{}_omega{:.3f}_dens{:.3f}_beta{:.4f}/'.format('renormalized' if self.renormalized else 'unrenormalized', self.nk, abs(self.tp), len(shape(self.band(1, params['t'], params['tp']))), self.g0, self.nw, self.omega, self.dens, self.beta)
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
        return -2.0*mean(G[:,:,-1]).real
    #---------------------------------------------------------------------------
    def compute_G(self, wn, ek, mu, S):
        #print('shape S', np.shape(S))
        return 1.0/(1j*wn[None,None,:] - (ek[:,:,None]-mu) - S)
    #---------------------------------------------------------------------------
    def compute_D(self, vn, PI):
        return 1.0/(-((vn**2)[None,None,:] + self.omega**2)/(2.0*self.omega) - PI)
    #---------------------------------------------------------------------------
    def compute_S(self, G, D):
        return -self.g0**2/self.nk**2 * conv(G, D, ['k-q,q','k-q,q'], [0,1], [True,True], self.beta)
    #---------------------------------------------------------------------------
    def compute_GG(self, G):
        return 2.0/self.nk**2 * conv(G, -G[:,:,::-1], ['k,k+q','k,k+q'], [0,1], [True,True], self.beta)
    #---------------------------------------------------------------------------
    def dyson_fermion(self, wn, ek, mu, S, axis):
        Sw, jumpS = fourier.t2w_fermion_alpha0(S, self.beta, axis)
        Gw = self.compute_G(wn, ek, mu, Sw)
        jumpG = -np.ones((self.nk, self.nk, 1))
        return fourier.w2t_fermion_alpha0(Gw, self.beta, axis, jumpG)
    #---------------------------------------------------------------------------
    def dyson_boson(self, vn, PI, axis):
        PIw = fourier.t2w_boson(PI, self.beta, axis)
        Dw  = self.compute_D(vn, PIw)
        return fourier.w2t_boson(Dw, self.beta, axis)

    #---------------------------------------------------------------------------
    def selfconsistency(self, sc_iter, frac=0.9, alpha=0.5, S0=None, PI0=None):
        savedir, wn, vn, ek, mu, deriv, dndmu = self.setup()
        
        print('\nSelfconsistency\n--------------------------')

        if S0 is None or PI0 is None:
            S  = zeros([self.nk,self.nk,self.ntau], dtype=complex)
            PI = zeros([self.nk,self.nk,self.ntau], dtype=complex)
        else:
            S  = S0
            PI = PI0

        change = [0, 0]
        for i in range(sc_iter):
            S0  = S[:]
            PI0 = PI[:]

            # compute Gtau, Dtau

            G = self.dyson_fermion(wn, ek, mu, S, 2)
            D = self.dyson_boson(vn, PI, 2)


            #figure()
            #plot(mean(G, axis=(0,1)).real)
            #savefig('Gtau')
            #exit()

            n = self.compute_n(G)
            mu -= alpha*(n-self.dens)/dndmu
            #mu = optimize.fsolve(lambda mu : self.compute_fill(self.compute_G(wn,ek,mu,S))-self.dens, mu)[0]

            # compute new selfenergy

            S  = self.compute_S(G, D)

            #figure()
            #plot(mean(S, axis=(0,1)).real)
            #plot(mean(S, axis=(0,1)).imag)
            #savefig('S')
            #exit()

            change[0] = mean(abs(S-S0))/(mean(abs(S+S0))+1e-10)
            S  = frac*S + (1-frac)*S0

            GG = self.compute_GG(G)
            PI = self.g0**2 * GG
            change[1] = mean(abs(PI-PI0))/(mean(abs(PI+PI0))+1e-10)
            PI = frac*PI + (1-frac)*PI0

            #if i%10==0:
            print('change = {:.3e}, {:.3e} and fill = {:.13f} mu = {:.5f} EF = {:.5f}'.format(change[0], change[1], n, mu, ek[self.nk//2, self.nk//2]-mu))

            #if params['g0']<1e-10: break

            if i>10 and change[0]<1e-14 and change[1]<1e-14 and abs(self.dens-n)<1e-5: break


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
        
        return savedir, G, D, S, GG
    #---------------------------------------------------------------------------
    def susceptibilities(self, sc_iter, G, D, GG, frac=0.8): 
        print('\nComputing Susceptibilities\n--------------------------')

        # convert to imaginary frequency
        G  = fourier.t2w(G, self.beta, 2, 'fermion')[0]
        D  = fourier.t2w(D, self.beta, 2, 'boson')
        GG = fourier.t2w(GG, self.beta, 2, 'boson')

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

        jumpx = np.zeros((self.nk, self.nk, 1))
        # momentum and frequency convolution 
        # the jump for F0 is zero
        jumpF0 = np.zeros((self.nk, self.nk, 1))
        jumpD  = None
        x0 = -self.g0**2/self.nk**2 * conv(F0, D, ['k-q,q','k-q,q', 'm,n-m'], [0,1,2], [True,True,False], self.beta, kinds=('fermion','boson','fermion'), op='...,...', jumps=(jumpF0, jumpD))

        jumpF0x = np.zeros((self.nk, self.nk, 1), dtype=complex)

        iteration = 0
        x = np.zeros((self.nk,self.nk,self.nw), dtype=complex)
        while iteration < sc_iter:
            x_initial = x.copy()

            # compute jumpF0x
            F0x_tau = fourier.w2t(F0*x, self.beta, 2, 'fermion', jumpF0x)
            jumpF0x = F0x_tau[:,:,0] + F0x_tau[:,:,-1]
            jumpF0x = jumpF0x[:,:,None]

            x = x0 - self.g0**2/self.nk**2 * conv(F0*x, D, ['k-q,q','k-q,q','m,n-m'], [0,1,2], [True,True,False], self.beta, kinds=('fermion','boson','fermion'), op='...,...', jumps=(jumpF0x, None))

            change = mean(abs(x - x_initial))/(mean(abs(x + x_initial))+1e-10)

            x = frac*x + (1-frac)*x_initial
            
            iteration += 1

            if change < 1e-10:
                break

            if iteration%max(sc_iter//20,1)==0:
                print(f'change {change:.4e}')

        print(f'change {change:.4e}')

        if change>1e-5:
            print('Susceptibility failed to converge')
            return None, None
    
        Xsc = 1.0 / (self.beta * self.nk**2) * 2.0*sum(F0*(1+x)).real
        print(f'Xsc {Xsc:.4f}')

        # compute the CDW susceptibility
        X0 = -GG[:,:,0]
        Xcdw = real(X0/(1.0 - 2.0*self.g0**2/self.omega * X0))

        Xcdw = ravel(Xcdw)
        a = argmax(abs(Xcdw))
        print('Xcdw = %1.4f'%Xcdw[a])

        if Xsc<0.0 or any(Xcdw<0.0): 
            print('Xcdw blew up')
            return None, None

        return Xsc, Xcdw


if __name__=='__main__':

    # example usage as follows :

    print('2D Renormalized Migdal')

    basedir = '/scratch/users/bln/elph/imagaxis/example/'
    if not os.path.exists(basedir): os.mkdir(basedir)

    params['beta'] = 16.0
    
    lamb = 0.00
    W    = 8.0
    params['g0'] = lamb2g0(lamb, params['omega'], W)
    print('g0 is ', params['g0'])
    
    migdal = Migdal(params, basedir)

    sc_iter = 60
    S0, PI0  = None, None
    savedir, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, frac=0.2)
    PI = params['g0']**2 * GG
    save(savedir + 'S.npy', S)
    save(savedir + 'PI.npy', PI)
    save(savedir + 'G.npy', G)
    save(savedir + 'D.npy', D)

    sc_iter = 60
    Xsc, Xcdw = migdal.susceptibilities(sc_iter, G, D, GG, frac=0.4)
    save(savedir + 'Xsc.npy',  [Xsc])
    save(savedir + 'Xcdw.npy', [Xcdw])

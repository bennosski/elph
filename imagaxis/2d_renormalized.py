from numpy import *
import time
from convolution import conv
import os
import sys
from scipy import optimize
from params import params

class Migdal:
    #---------------------------------------------------------------------------
    def __init__(self, params):
        for key in params:
            setattr(self, key, params[key])
    #---------------------------------------------------------------------------
    def myp(x):
        print(mean(abs(x.real)), mean(abs(x.imag)))
    #---------------------------------------------------------------------------
    def setup(self):
        print('\nParameters\n----------------------------')

        print('Nw     = {}'.format(self.Nw))
        print('Nk     = {}'.format(self.Nk))
        print('beta   = {:.3f}'.format(self.beta))
        print('omega  = {}'.format(self.omega))
        print('g0     = {:.3f}'.format(self.g0))
        print('dens   = {}'.format(self.dens))
        print('renorm = {}'.format(self.renormalized))
        print('SC     = {}'.format(self.sc))
        print('dim    = {}'.format(len(shape(self.band(1)))))     
        savedir = 'data/data_renormalized_Nk{}_dim{}_g0{:.5f}_Nw{}_omega{:.3f}_dens{:.3f}/'.format(self.Nk, len(shape(self.band(1))), self.g0, self.Nw, self.omega, self.dens)
        if not os.path.exists('data/'): os.mkdir('data/')
        if not os.path.exists(savedir): os.mkdir(savedir)

        assert self.Nk%2==0
        assert self.Nw%2==0

        wn = pi/self.beta * (2*arange(self.Nw)+1)
        vn = pi/self.beta * (2*arange(self.Nw+1))

        ek = self.band(self.Nk)

        # estimate filling and dndmu at the desired filling
        mu = optimize.fsolve(lambda mu : 2.0*mean(1.0/(exp(self.beta*(ek-mu))+1.0))-self.dens, 0.0)
        deriv = lambda mu : 2.0*mean(-self.beta*exp(self.beta*(ek-mu))/(exp(self.beta*(ek-mu))+1.0)**2)
        dndmu = deriv(mu)

        print('mu optimized = %1.3f'%mu)
        print('dndmu = %1.3f'%dndmu)

        return savedir, wn, vn, ek, mu, deriv, dndmu
    #---------------------------------------------------------------------------
    def compute_fill(self, G):
        return 1.0 + 2.0/(self.beta * self.Nk**2) * sum(G, axis=None).real
    #---------------------------------------------------------------------------
    def compute_G(self, wn, ek, mu, S):
        return 1.0/(1j*wn[None,None,:] - (ek[:,:,None]-mu) - S)
    #---------------------------------------------------------------------------
    def compute_D(self, vn, PI):
        return 1.0/(-((vn**2)[None,None,:] + self.omega**2)/(2.0*self.omega) - PI)
    #---------------------------------------------------------------------------
    def compute_S(self, G, D):
        return -self.g0**2/self.Nk**2 * conv(G, D, ['k-q,q','k-q,q','m,n-m'], [0,1,2], [True,True,False], self.beta, kinds=('fermion','boson','fermion'))
    #---------------------------------------------------------------------------
    def compute_PI(self, G):
        return 2.0*self.g0**2/self.Nk**2 * conv(G, G, ['k,k+q','k,k+q','m,m+n'], [0,1,2], [True,True,False], self.beta, kinds=('fermion','fermion','boson'))
    #---------------------------------------------------------------------------
    def selfconsistency(self, sc_iter, frac=0.9, alpha=0.5, S0=None, PI0=None):
        savedir, wn, vn, ek, mu, deriv, dndmu = self.setup()

        print('\nSelfconsistency\n--------------------------')

        if S0 is None or PI0 is None:
            S  = zeros([self.Nk,self.Nk,self.Nw], dtype=complex)
            PI = zeros([self.Nk,self.Nk,self.Nw+1], dtype=complex)
        else:
            S  = S0
            PI = PI0

        G = self.compute_G(wn, ek, mu, S)
        D = self.compute_D(vn, PI) 

        change = [0, 0]
        for i in range(sc_iter):
            S0  = S[:]
            PI0 = PI[:]

            S  = self.compute_S(G, D)
            change[0] = mean(abs(S-S0))/mean(abs(S+S0))
            S  = frac*S + (1-frac)*S0

            PI = self.compute_PI(G)
            change[1] = mean(abs(PI-PI0))/mean(abs(PI+PI0))
            PI = frac*PI + (1-frac)*PI0

            G = self.compute_G(wn, ek, mu, S)
            D = self.compute_D(vn, PI) 

            n = self.compute_fill(G)
            mu += alpha*(n-self.dens)/dndmu

            print('change = %1.3e, %1.3e and fill = %.13f'%(change[0], change[1], self.compute_fill(G)))

            if i>10 and change[0]<1e-14 and change[1]<1e-14: break

        save(savedir+'Nw',    [self.Nw])
        save(savedir+'Nk',    [self.Nk])
        save(savedir+'beta',  [self.beta])
        save(savedir+'omega', [self.omega])
        save(savedir+'g0',    [self.g0])
        save(savedir+'dens',  [self.dens])
        save(savedir+'renormalized', [self.renormalized])
        save(savedir+'sc',    [self.sc])

        return savedir, G, D, S, PI
    #---------------------------------------------------------------------------
    def susceptibilities(self, sc_iter, G, D, PI, frac=0.9): 
        print('\nComputing Susceptibilities\n--------------------------')

        F0 = G * conj(G)
        T  = ones([self.Nk,self.Nk,self.Nw])

        change = 1
        iteration = 0
        while iteration < sc_iter:
            T0 = T.copy()

            T = conv(F0*T, D, ['k-q,q','k-q,q','m,n-m'], [0,1,2], [True,True,False], self.beta, kinds=('fermion','boson','fermion')) 
            T *= -self.g0**2/(self.Nk**2) 
            T += 1.0

            T = frac*T + (1-frac)*T0

            change = mean(abs(T-T0))/mean(abs(T+T0))
            print(f'change = {change:.5e}')

            iteration += 1
            
            if change < 1e-10:
                break
            
        Xsc = 1.0/(self.Nk**2*self.beta) * real(sum(F0*T))
        Xsc = 2.0*Xsc # factor of two because need to sum negative frequencies as well

        print('Xsc = %1.4f'%real(Xsc))

        # compute the CDW susceptibility
        #X0 = -PI[:,:,Nw//2]/alpha**2
        #Xcdw = real(X0/(1.0 - alpha**2/omega**2 * X0))

        X0 = -PI[:,:,0]/self.g0**2
        Xcdw = real(X0/(1.0 - 2.0*self.g0**2/self.omega * X0))

        Xcdw = ravel(Xcdw)
        a = argmax(abs(Xcdw))
        print('Xcdw = %1.4f'%Xcdw[a])

        if Xsc<0.0 or any(Xcdw<0.0): 
            return None, None

        return Xsc, Xcdw
    #---------------------------------------------------------------------------
    def doubled_susceptibilities(self, G, D, PI, frac=0.9): 

        # compute susceptibilities

        print('shapeG', shape(G))
        print('shapeD', shape(D))

        # extend to + and - freqs
        Nw = 2*self.Nw
        G_ = concatenate((conj(G[:, :, ::-1]), G), axis=-1)
        D_ = concatenate((conj(D[:, :, :0:-1]), D), axis=-1)

        print('shapeG_', shape(G_))
        print('shapeD_', shape(D_))

        F0 = G_ * G_[:,:,::-1]
        T  = ones([self.Nk,self.Nk,Nw])

        tmp = zeros([self.Nk,self.Nk,2*Nw], dtype=complex)
        tmp[:,:,:Nw+1] = D_
        tmp = fft.fftn(tmp)

        change = 1
        iteration = 0
        while change > 1e-10:
            T0 = T.copy()

            m = zeros([self.Nk,self.Nk,2*Nw], dtype=complex)
            m[:,:,:Nw] = F0*T
            m = fft.fftn(m)
            T = fft.ifftn(m * tmp)
            T = roll(T, (-self.Nk//2, -self.Nk//2, -Nw//2), axis=(0,1,2))[:,:,:Nw]
            T *= -self.g0**2/(self.beta*self.Nk**2) 
            T += 1.0

            change = mean(abs(T-T0))/mean(abs(T+T0))
            if iteration%100==0: print('change : %1.3e'%change)

            T = frac*T + (1-frac)*T0

            iteration += 1
            if iteration>2000: exit()

        Xsc = 1.0/(self.Nk**2*self.beta) * real(sum(F0*T))
        #save(savedir+'Xsc.npy', [Xsc])
        print('Xsc = %1.4f'%real(Xsc))

        # self.compute the CDW susceptibility
        #X0 = -PI[:,:,Nw//2]/alpha**2
        #Xcdw = real(X0/(1.0 - alpha**2/omega**2 * X0))

        X0 = -PI[:,:,0]/self.g0**2
        Xcdw = real(X0/(1.0 - 2.0*self.g0**2/self.omega * X0))
        #save(savedir+'Xcdw.npy', Xcdw)

        Xcdw = ravel(Xcdw)
        a = argmax(abs(Xcdw))
        print('Xcdw = %1.4f'%Xcdw[a])

        if Xsc<0.0 or any(Xcdw<0.0): 
            return None, None

        return Xsc, Xcdw
#---------------------------------------------------------------------------        

if __name__=='__main__':
    
    print('2D Renormalized Migdal')

    migdal = Migdal(params)

    sc_iter = 10
    S0, PI0  = None, None
    savedir, G, D, S, PI = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0)
    save(savedir + 'S.npy', S)
    save(savedir + 'PI.npy', PI)

    sc_iter = 10
    Xsc, Xcdw = migdal.susceptibilities(sc_iter, G, D, PI)
    save(savedir + 'Xsc.npy',  [Xsc])
    save(savedir + 'Xcdw.npy', [Xcdw])


    """    
    dbeta = 2.0

    Xscs = []

    while dbeta >= 0.25:

        print(f'\nbeta = {beta:.4f}')

        G, D, S, PI = selfconsistency(S0, PI0)

        #Xsc, Xcdw = susceptibilities(G, PI)
        
        if Xsc is None or Xcdw is None:
            dbeta /= 2.0
        else:
            Xscs.append([beta, Xsc])
            S0, PI0 = S, PI

        params.beta += dbeta
    
        #save(savedir + 'Xscs.npy', array(Xscs))
        
    """


from numpy import *
import time
from convolution import conv
import os
import sys
from scipy import optimize
from params import params
import fourier

#import matplotlib
#matplotlib.use('agg')
#from matplotlib.pyplot import *

class Migdal:
    #---------------------------------------------------------------------------
    def __init__(self, params):
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
        print('dens   = {}'.format(self.dens))
        print('renorm = {}'.format(self.renormalized))
        print('SC     = {}'.format(self.sc))
        print('dim    = {}'.format(len(shape(self.band(1)))))     
        savedir = 'data/data_renormalized_nk{}_dim{}_g0{:.5f}_nw{}_omega{:.3f}_dens{:.3f}/'.format(self.nk, len(shape(self.band(1))), self.g0, self.nw, self.omega, self.dens)
        if not os.path.exists('data/'): os.mkdir('data/')
        if not os.path.exists(savedir): os.mkdir(savedir)

        assert self.nk%2==0
        assert self.nw%2==0

        self.ntau = 2*self.nw + 1

        wn = (2*arange(self.nw)+1) * pi / self.beta
        vn = (2*arange(self.nw+1)) * pi / self.beta
        
        ek = self.band(self.nk)

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
    def compute_G(self, wn, ek, mu, S):
        return 1.0/(1j*wn[None,None,:] - (ek[:,:,None]-mu) - S)
    #---------------------------------------------------------------------------
    def compute_D(self, vn, PI):
        return 1.0/(-((vn**2)[None,None,:] + self.omega**2)/(2.0*self.omega) - PI)
    #---------------------------------------------------------------------------
    def compute_S(self, G, D):
        return -self.g0**2/self.nk**2 * conv(G, D, ['k-q,q','k-q,q'], [0,1], [True,True], self.beta)
    #---------------------------------------------------------------------------
    def compute_PI(self, G):
        #return 2.0*self.g0**2/self.nk**2 * conv(G, G, ['k,k+q','k,k+q','m,m+n'], [0,1,2], [True,True,False], self.beta, kinds=('fermion','fermion','boson'))
        return 2.0*self.g0**2/self.nk**2 * conv(G, -G[:,:,::-1], ['k,k+q','k,k+q'], [0,1], [True,True], self.beta)
    #---------------------------------------------------------------------------
    def dyson_fermion(self, wn, ek, mu, S, axis):
        Sw = apply_along_axis(fourier.t2w_fermion_alpha0, axis, S, self.beta)
        Gw = self.compute_G(wn, ek, mu, Sw)
        return apply_along_axis(fourier.w2t_fermion_alpha0, axis, Gw, self.beta)

    def dyson_boson(self, vn, PI, axis):
        PIw = apply_along_axis(fourier.t2w_boson, axis, PI, self.beta)
        Dw  = self.compute_D(vn, PIw)
        return apply_along_axis(fourier.w2t_boson, axis, Dw, self.beta)

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

            n = -2.0*mean(G[:,:,-1]).real
            mu -= alpha*(n-self.dens)/dndmu
            #mu = optimize.fsolve(lambda mu : self.compute_fill(self.compute_G(wn,ek,mu,S))-self.dens, mu)[0]

            # compute new selfenergy

            S  = self.compute_S(G, D)
            change[0] = mean(abs(S-S0))/mean(abs(S+S0))
            S  = frac*S + (1-frac)*S0

            PI = self.compute_PI(G)
            change[1] = mean(abs(PI-PI0))/mean(abs(PI+PI0))
            PI = frac*PI + (1-frac)*PI0

            print('change = {:.3e}, {:.3e} and fill = {:.13f} mu = {:.5f} EF = {:.5f}'.format(change[0], change[1], n, mu, ek[self.nk//2, self.nk//2]-mu))

            if i>10 and change[0]<1e-14 and change[1]<1e-14: break

        save(savedir+'nw',    [self.nw])
        save(savedir+'nk',    [self.nk])
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

        # convert to imaginary frequency
        G  = apply_along_axis(fourier.t2w, 2,  G, self.beta, 'fermion')
        D  = apply_along_axis(fourier.t2w, 2,  D, self.beta, 'boson')
        PI = apply_along_axis(fourier.t2w, 2, PI, self.beta, 'boson')

        F0 = G * conj(G)
        T  = ones([self.nk,self.nk,self.nw])

        change = 1
        iteration = 0
        while iteration < sc_iter:
            T0 = T.copy()

            T = conv(F0*T, D, ['k-q,q','k-q,q','m,n-m'], [0,1,2], [True,True,False], self.beta, kinds=('fermion','boson','fermion')) 
            T *= -self.g0**2/(self.nk**2) 
            T += 1.0

            T = frac*T + (1-frac)*T0

            change = mean(abs(T-T0))/mean(abs(T+T0))
            print('change = {:.5e}'.format(change))

            iteration += 1
            
            if change < 1e-10:
                break
            
        #FT = F0*T
        #Xsc = 1.0/(self.beta * self.nk**2) * real(sum(FT[:,:,0]).real + 2.0*sum(FT[:,:,1:]).real)
        Xsc = 1.0 / (self.beta * self.nk**2) * 2.0*sum(F0*T).real
        print('Xsc = %1.4f'%Xsc)

        Xsc2 = 1.0 / self.nk**2 * sum(F0*T, axis=(0,1))
        Xsc2 = fourier.w2t(Xsc2, self.beta, 'fermion')[0].real
        print('Xsc2 = %1.4f'%Xsc2)

        # compute the CDW susceptibility
        #X0 = -PI[:,:,nw//2]/alpha**2
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
        nw = 2*self.nw
        G_ = concatenate((conj(G[:, :, ::-1]), G), axis=-1)
        D_ = concatenate((conj(D[:, :, :0:-1]), D), axis=-1)

        print('shapeG_', shape(G_))
        print('shapeD_', shape(D_))

        F0 = G_ * G_[:,:,::-1]
        T  = ones([self.nk,self.nk,nw])

        tmp = zeros([self.nk,self.nk,2*nw], dtype=complex)
        tmp[:,:,:nw+1] = D_
        tmp = fft.fftn(tmp)

        change = 1
        iteration = 0
        while change > 1e-10:
            T0 = T.copy()

            m = zeros([self.nk,self.nk,2*nw], dtype=complex)
            m[:,:,:nw] = F0*T
            m = fft.fftn(m)
            T = fft.ifftn(m * tmp)
            T = roll(T, (-self.nk//2, -self.nk//2, -nw//2), axis=(0,1,2))[:,:,:nw]
            T *= -self.g0**2/(self.beta*self.nk**2) 
            T += 1.0

            change = mean(abs(T-T0))/mean(abs(T+T0))
            if iteration%100==0: print('change : %1.3e'%change)

            T = frac*T + (1-frac)*T0

            iteration += 1
            if iteration>2000: exit()

        Xsc = 1.0/(self.nk**2*self.beta) * real(sum(F0*T))
        #save(savedir+'Xsc.npy', [Xsc])
        print('Xsc = %1.4f'%real(Xsc))

        # self.compute the CDW susceptibility
        #X0 = -PI[:,:,nw//2]/alpha**2
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
    
    lamb = 0.6
    W    = 8.0
    params['g0'] = sqrt(0.5 * lamb / 2.4 * params['omega'] * W)
    print('g0 is ', params['g0'])
    
    #params['g0'] = 0.238
    #params['g0'] = 0.0

    migdal = Migdal(params)

    sc_iter = 300
    S0, PI0  = None, None
    savedir, G, D, S, PI = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, frac=0.2)
    save(savedir + 'S.npy', S)
    save(savedir + 'PI.npy', PI)
    save(savedir + 'G.npy', G)
    save(savedir + 'D.npy', D)

    sc_iter = 300
    Xsc, Xcdw = migdal.susceptibilities(sc_iter, G, D, PI, frac=0.7)
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


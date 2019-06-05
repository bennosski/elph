from numpy import *
import numpy as np
import time
from convolution import conv
import os
import sys
from scipy import optimize
from params import params

savedir = None

def myp(x):
    print(mean(abs(x.real)), mean(abs(x.imag)))

def band(kxs, kys):
    return -2.0*(cos(kxs) + cos(kys))  #+ alpha**2
    #return -2.0*(cos(kxs) + cos(kys)) + 4.0*0.3*cos(kxs)*cos(kys)

def init():
    Nw, Nk, beta, omega, lamb, W, g0, dens = params.Nw, params.Nk, params.beta, params.omega, params.lamb, params.W, params.g0, params.dens
    
    print(f'beta {beta:.3f}')
    print(f'Nk {Nk}')
    print(f'g0 = {g0:.3f}')
    print(f'lamb = {lamb:.3f}')
    
    #savedir = f'data/data_renormalized_{Nk}b{Nk}_lamb{lamb:.3f}_beta{beta:.1f}/' #%(Nk,Nk,lamb,beta)
    savedir = f'data/data_renormalized_{Nk}b{Nk}_lamb{lamb:.3f}_Nw{Nw}_omega{omega}_dens{dens:.3f}/'
    if not os.path.exists('data/'): os.mkdir('data/')
    if not os.path.exists(savedir): os.mkdir(savedir)

    wn = pi/beta * (2*arange(Nw)+1)
    vn = pi/beta * (2*arange(Nw+1))
    
    kys, kxs = meshgrid(arange(-pi, pi, 2*pi/Nk), arange(-pi, pi, 2*pi/Nk))

    ek = band(kxs, kys)

    assert Nk%2==0

    # estimate filling and dndmu at the desired filling
    mu = optimize.fsolve(lambda mu : 2.0*mean(1.0/(exp(beta*(ek-mu))+1.0))-dens, 0.0)
    deriv = lambda mu : 2.0*mean(-beta*exp(beta*(ek-mu))/(exp(beta*(ek-mu))+1.0)**2)
    dndmu = deriv(mu)

    print('mu optimized = %1.3f'%mu)
    print('dndmu = %1.3f'%dndmu)

    return savedir, wn, vn, ek, mu, deriv, dndmu

def compute_fill(G):
    beta, Nk = params.beta, params.Nk
    return 1.0 + 2.0/(beta * Nk**2) * np.sum(G).real

def compute_S(G, D):
    g0, beta, Nk, Nw = params.g0, params.beta, params.Nk, params.Nw
    return -g0**2/Nk**2 * conv(G, D, ['k-q,q','k-q,q','m,n-m'], [0,1,2], [True,True,False], params, kinds=('fermion','boson','fermion'))

def compute_PI(G):
    g0, beta, Nk, Nw = params.g0, params.beta, params.Nk, params.Nw    
    return 2.0*params.g0**2/Nk**2 * conv(G, G, ['k,k+q','k,k+q','m,m+n'], [0,1,2], [True,True,False], params, kinds=('fermion','fermion','boson'))

def compute_G(wn, ek, mu, S):
    return 1.0/(1j*wn[None,None,:] - (ek[:,:,None]-mu) - S)

def compute_D(vn, omega, PI):
    return 1.0/(-((vn**2)[None,None,:] + omega**2)/(2.0*omega) - PI)
    

def selfconsistency(S0=None, PI0=None):
    Nw, Nk, beta, omega, lamb, W, g0, dens = params.Nw, params.Nk, params.beta, params.omega, params.lamb, params.W, params.g0, params.dens    
    
    savedir, wn, vn, ek, mu, deriv, dndmu = init()

    if S0 is None or PI0 is None:
        S  = zeros([Nk,Nk,Nw], dtype=complex)
        PI = zeros([Nk,Nk,Nw+1], dtype=complex)
    else:
        S  = S0
        PI = PI0

    G = compute_G(wn, ek, mu, S)
    D = compute_D(vn, omega, PI) 
    
    # solve Matsubara piece
    print('\n Solving Matsubara piece')

    change = [0, 0]
    frac = 0.9
    for i in range(100):
        S0  = S[:]
        PI0 = PI[:]

        S  = compute_S(G, D)
        change[0] = mean(abs(S-S0))/mean(abs(S+S0))
        S  = frac*S + (1-frac)*S0

        PI = compute_PI(G)
        change[1] = mean(abs(PI-PI0))/mean(abs(PI+PI0))
        PI = frac*PI + (1-frac)*PI0

        G = compute_G(wn, ek, mu, S)
        D = compute_D(vn, omega, PI) 

        n = compute_fill(G)
        mu += 0.5*(n-dens)/dndmu

        print('change = %1.3e, %1.3e and fill = %.13f'%(change[0], change[1], compute_fill(G)))

        if i>10 and change[0]<1e-15 and change[1]<1e-15: break

        #if i%10==0:
        #    save(savedir+'S', S)
        #    save(savedir+'PI', PI)

    save(savedir+'wn', wn)
    save(savedir+'vn', vn)
    save(savedir+'Nk', [Nk])
    save(savedir+'lamb', [lamb])
    save(savedir+'omega', [omega])

    return G, D, S, PI

def susceptibilities(G, PI): 
    Nw, Nk, beta, omega, lamb, W, g0, dens = params.Nw, params.Nk, params.beta, params.omega, params.lamb, params.W, params.g0, params.dens

    # compute susceptibilities

    F0 = G * G[:,:,::-1]
    T  = ones([Nk,Nk,Nw])

    tmp = zeros([Nk,Nk,2*Nw], dtype=complex)
    tmp[:,:,:Nw+1] = D
    tmp = fft.fftn(tmp)

    change = 1
    iteration = 0
    frac = 0.6
    while change > 1e-10:
        T0 = T.copy()

        m = zeros([Nk,Nk,2*Nw], dtype=complex)
        m[:,:,:Nw] = F0*T
        m = fft.fftn(m)
        T = fft.ifftn(m * tmp)
        T = roll(T, (-Nk//2, -Nk//2, -Nw//2), axis=(0,1,2))[:,:,:Nw]
        T *= -g0**2/(beta*Nk**2) 
        T += 1.0

        change = mean(abs(T-T0))/mean(abs(T+T0))
        if iteration%100==0: print('change : %1.3e'%change)

        T = frac*T + (1-frac)*T0

        iteration += 1
        if iteration>2000: exit()

    Xsc = 1.0/(Nk**2*beta) * real(sum(F0*T))
    #save(savedir+'Xsc.npy', [Xsc])
    print('Xsc = %1.4f'%real(Xsc))

    # compute the CDW susceptibility
    #X0 = -PI[:,:,Nw//2]/alpha**2
    #Xcdw = real(X0/(1.0 - alpha**2/omega**2 * X0))
    X0 = -PI[:,:,Nw//2]/g0**2
    Xcdw = real(X0/(1.0 - 2.0*g0**2/omega * X0))
    #save(savedir+'Xcdw.npy', Xcdw)

    Xcdw = ravel(Xcdw)
    a = argmax(abs(Xcdw))
    print('Xcdw = %1.4f'%Xcdw[a])

    if Xsc<0.0 or any(Xcdw<0.0): return None, None

    return Xsc, Xcdw

if __name__=='__main__':
    

    '''
    savedir = None
    if len(sys.argv)>1:
        savedir = sys.argv[1]
        print('Using data from %s\n'%savedir)
    '''


    print('running renormalized ME')

    S0, PI0  = None, None
    G, D, S, PI = selfconsistency(S0, PI0)


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


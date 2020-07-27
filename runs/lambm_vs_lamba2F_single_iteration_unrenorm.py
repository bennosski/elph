

import sys
import os
import src
from migdal_2d import Migdal
from real_2d import RealAxisMigdal
from functions import band_square_lattice, mylamb2g0, lamb2g0_ilya, myg02lamb, g02lamb_ilya
import numpy as np
from interpolator import Interp
import matplotlib.pyplot as plt
import single_iter
from scipy.optimize import root_scalar
from scipy.interpolate import interp2d
from scipy.misc import derivative
import fourier



def kF(theta, theta0, x0, y0, t, tp, mu):
    def r2k(r):
        return x0 + r*np.cos(theta0+theta), y0 + r*np.sin(theta0+theta)
        
    def ek(r):
        kx, ky = r2k(r)
        return -2.0*t*(np.cos(kx)+np.cos(ky)) - 4.0*tp*np.cos(kx)*np.cos(ky) - mu

    r = root_scalar(ek, bracket=(0, np.pi)).root
    #print('found kx, ky = ', r, r2k(r))
    deriv = derivative(ek, r, dx=1e-4)
    assert deriv < 0
    return r2k(r), r, deriv


def corrected_kF(S, theta, theta0, x0, y0, t, tp, mu):
    I = S[len(S)//2] # this is SR(w=0, k)
    
    def r2k(r):
        return x0 + r*np.cos(theta0+theta), y0 + r*np.sin(theta0+theta)
        
    def ekpSR(r):
        kx, ky = r2k(r)
        ek = -2.0*t*(np.cos(kx)+np.cos(ky)) - 4.0*tp*np.cos(kx)*np.cos(ky) - mu
        return ek + I.real

    r = root_scalar(ekpSR, bracket=(0, np.pi)).root
    
    deriv = derivative(ekpSR, r, dx=1e-4)
    assert deriv < 0
    return r2k(r), r, deriv


# function to compute fermi velocity (need deriv sigma real)
def vel(kx, ky, dEdk, S, wr, nr):
    dSdE = (S[nr//2+2] - S[nr//2-2]).real / (wr[nr//2+2] - wr[nr//2-2])
    #assert dSdE < 0
    return np.abs(dEdk) / (1 - dSdE)


def corrected_dos(S, mu, wr, ntheta=5):
    
    #wr, nr, nk, SR, DR, mu, t, tp, g0, omega = load(basedir, folder)
    
    t = 1
    tp = -0.3
    nr = len(S)
    
    dtheta = np.pi/(2*ntheta)
    thetas = np.arange(dtheta/2, np.pi/2, dtheta)
    assert len(thetas) == ntheta
    
    corners = ((0,        -np.pi, -np.pi), 
               (-np.pi/2, -np.pi,  np.pi),
               (-np.pi,    np.pi,  np.pi),
               (np.pi/2,   np.pi, -np.pi))
    
    vels  = []
    rs    = []
    for corner in corners:
        for theta in thetas:
            #(kx, ky), r, dEdk =  corrected_kF(S, theta, corner[0], corner[1], corner[2], t, tp, mu)
            (kx, ky), r, dEdk =  kF(theta, corner[0], corner[1], corner[2], t, tp, mu)
            
            rs.append(r)
            v = vel(kx, ky, dEdk, S, wr, nr)
            vels.append(v)
    
    # compute normalization factor
    dos = 0
    for ik in range(ntheta):
        dos += rs[ik] / vels[ik] * dtheta
        
    dos = 4 * dos / (2*np.pi)**2
    #print('dos renormalized (unrenormalized version is 0.3) = ', dos)
    
    return dos
 

def test(beta, g0):
    
    #w = np.arange(-4, 4, 0.005)
    
    idelta = 1j/beta
    dw = 1/beta
    
    w = np.arange(-2.25, 2.25, dw)
    
    print('w[0] =', w[len(w)//2])
    assert np.abs(w[len(w)//2]) < 1e-10
    
    omega = 1.0
    #idelta = 0.010j
    idelta = idelta
    dens = 0.8
    
    nk = 400
    ek = band_square_lattice(nk, 1, -0.3)
    
    mu = single_iter.compute_mu(ek, beta, nk, dens)
    #mu = -1.09928783
    mu, S = single_iter.get_S(w, ek, g0, beta, omega, nk, idelta, dens, mu)

    print('filling', single_iter.compute_fill(ek, beta, mu, nk))
    
    print('shape S', np.shape(S))
    
    dos = 1/nk**2 * np.sum(1/(w[None,None,:] - (ek[:,:,None]-mu) - S[None,None,:] + idelta), axis=(0,1))
    plt.figure()
    plt.plot(w, -1.0/np.pi*dos.imag)
    plt.title('dos')
    
    
    plt.figure()
    plt.plot(w, S.real)
    plt.plot(w, S.imag)
    print('self-energy from single iteration formula')
    
    print('dos unrenorm', corrected_dos([0,0,0,0,0], mu, w, ntheta=80))
    
    S = S - S[len(w)//2].real
    dos = corrected_dos(S, mu, w, ntheta=200)
    print('dos renorm', dos)
    
    lamba2F = 2*g0**2 * dos / omega
    
    nw = len(w)
    lambm = - (S[nw//2+2] - S[nw//2-2]).real / (w[nw//2+2] - w[nw//2-2])

    print('lamba2F', lamba2F)
    print('lamb m', lambm)
 
    np.save('data/w', w)
    np.save('data/lambm', [lambm])
    np.save('data/lamba2F', [lamba2F])
    np.save('data/S', S)
    
    
 


def plot():
    w = np.load('data/w.npy')
    S = np.load('data/S.npy')
    h = len(w)//2
    S = S - S[h].real
    
    plt.figure()
    plt.plot(w, -S.real)
    #plt.plot(w, S.imag)
    
    plt.figure()
    
    wm = np.concatenate((w[0:len(w)//2], w[len(w)//2+1:]))
    Sm = np.concatenate((S.real[0:len(w)//2], S.real[len(w)//2+1:]))
    plt.plot(wm, -Sm/wm)
    
    lambnsc = -Sm[h]/wm[h]
    
    print('lambm not self-consistent', lambnsc)
    
    #folder = '/scratch/users/bln/elph/data/2dfixedn/data/data_unrenormalized_nk150_abstp0.300_dim2_g00.63246_nw128_omega1.000_dens0.800_beta16.0000_QNone'
    folder = '/scratch/users/bln/elph/data/2dfixedn/data/data_unrenormalized_nk150_abstp0.300_dim2_g01.26491_nw128_omega1.000_dens0.800_beta16.0000_QNone'


    w = np.load(folder + '/w.npy')
    S = np.load(folder + '/SR.npy')[0,0]
    h = len(w)//2
    S -= S.real[h]
    wm = np.concatenate((w[:h], w[h+1:]))
    Sm = np.concatenate((S[:h], S[h+1:]))
    
    plt.plot(wm, -Sm/wm)
    
    lambsc = -Sm[h]/wm[h]
    print('lambda m selfconsistent', lambsc)
    
    plt.xlim(0, 2)
    #plt.ylim(-0.05, 0.27)
    plt.ylim(-0.1, 1.05)
    
    
    plt.legend(['single-iteration (-Re $\Sigma$)/$\omega$', 'self-consistent (-Re $\Sigma$)/$\omega$'], fontsize=10)
    plt.xlabel('$\omega$', fontsize=14)

    plt.savefig('data/sigma_and_deriv_0p4')    
    
    return lambnsc, lambsc

#plot()
    
    
def compute_rhoEF():
    w = np.load('data/w.npy')
    S = np.load('data/S.npy')
    
    nk = 400
    ek = band_square_lattice(nk, 1, -0.3)
    ek = ek[:,:,None]
    mu = -1.09928783
    idelta = 0.001j
    beta = 20
    
    print('van hove at', ek[0,nk//2]-mu)
    
    h = len(w)//2 
    #S = S - S.real[h]
    
    S = S - S[h].real
    G =  1/nk**2 * np.sum(1/(w[None,None,:] - (ek-mu) - 10*S[None,None,:] + idelta), axis=(0,1))
    A = -1/np.pi * G.imag
    
    dw = (w[-1]-w[0])/(len(w)-1)
    nF = 1/(np.exp(w[None,None,:]*beta) + 1)
    fill = np.trapz(2*A*nF, dx=dw)
    print('filling = ', fill)
    
    
    plt.figure()
    plt.plot(w, A)
    
   
    print('rho EF', A[h])
    
    print('amax A', np.amax(A))
    print('max w ', w[np.argmax(A)])
    
#compute_rhoEF()


def main():
    #g0 = lamb2g0_ilya(0.1, 1, 8)
    g0 = lamb2g0_ilya(0.4, 1, 8)
    
    
    print('g0', g0)
    print('mylamb', myg02lamb(g0, 1, 8))
    print('lamb ilya', g02lamb_ilya(g0, 1, 8))

    test(16, g0)
    plot()
    
 
def main_vs_T():
    g0 = lamb2g0_ilya(0.1, 1, 8)
    print('g0', g0)
    print('mylamb', myg02lamb(g0, 1, 8))
    print('lamb ilya', g02lamb_ilya(g0, 1, 8))

    
    
main()



# -------------------------------------------------------
# 1D case


def test_1d(beta, g0):
    
    #w = np.arange(-4, 4, 0.005)
    w = np.arange(-4, 4, 0.010)
    
    omega = 1.0
    idelta = 0.020j

    nk = int(1000*w[-1])
    ek = np.linspace(w[0]/2, w[-1]/2, nk)
    mu = 0
    mu, S = single_iter.get_S_constant_N(w, ek, g0, beta, omega, nk, idelta, None, mu)
        
    
    print('filling', single_iter.compute_fill(w, ek, beta, mu, nk))
    
    plt.figure()
    plt.plot(w, S.real)
    plt.plot(w, S.imag)
    
    print('dos unrenorm', corrected_dos([0,0,0,0,0], mu, w, ntheta=80))
    
    S = S - S[len(w)//2].real
    dos = corrected_dos(S, mu, w, ntheta=200)
    print('dos renorm', dos)
   
    lamba2F = 2*g0**2 * dos / omega
    
    nw = len(w)
    lambm = - (S[nw//2+2] - S[nw//2-2]).real / (w[nw//2+2] - w[nw//2-2])

    print('lamba2F', lamba2F)
    print('lamb m', lambm)
 
    np.save('data/w', w)
    np.save('data/lambm', [lambm])
    np.save('data/lamba2F', [lamba2F])
    np.save('data/S', S)
    

def compute_rhoEF_1d():
    w = np.load('data/w.npy')
    S = np.load('data/S.npy')
    
    nk = int(1000*w[-1])
    ek = np.linspace(w[0]/2, w[-1]/2, nk)
    mu = 0
    idelta = 0.005j
    beta = 50
    
    ek = ek[:,None]
    
    h = len(w)//2 

    
    G =  1/nk * np.sum(1/(w[None,:] - (ek-mu) + idelta), axis=(0))
    A = -1/np.pi * G.imag
    
    dw = (w[-1]-w[0])/(len(w)-1)
    nF = 1/(np.exp(w[None,:]*beta) + 1)
    fill = np.trapz(2*A*nF, dx=dw)
    print('filling = ', fill)
    
    plt.figure()
    plt.plot(w, A)
    plt.title('bare DOS')
    
   
    print('rho EF', A[h])
    
    
    S = S - S[h].real
    G =  1/nk * np.sum(1/(w[None,:] - (ek-mu) - S[None,:] + idelta), axis=(0))
    A = -1/np.pi * G.imag
    
    dw = (w[-1]-w[0])/(len(w)-1)
    nF = 1/(np.exp(w[None,:]*beta) + 1)
    fill = np.trapz(2*A*nF, dx=dw)
    print('filling = ', fill)
    
    
    plt.figure()
    plt.plot(w, A)
    plt.title('interacting DOS')
   
    print('rho EF', A[h])
    
    print('amax A', np.amax(A))
    print('max w ', w[np.argmax(A)])


def plot_1d():
    w = np.load('data/w.npy')
    S = np.load('data/S.npy')
    h = len(w)//2
    S = S - S[h].real
    
    plt.figure()
    plt.plot(w, -S.real)

    
    #plt.plot(w, S.imag)
    
    wm = np.concatenate((w[0:len(w)//2], w[len(w)//2+1:]))
    Sm = np.concatenate((S.real[0:len(w)//2], S.real[len(w)//2+1:]))
    plt.plot(wm, -Sm/wm)
    #plt.xlim(0, 2)
    #plt.ylim(-0.5, 1.6)
    plt.legend(['-Re $\Sigma$', '(-Re $\Sigma$)/$\omega$'], fontsize=13)
    plt.xlabel('$\omega$', fontsize=14)



    lamb = 1
    omega = 1
    Sexact = lamb * omega * np.log(np.abs((omega-w)/(w+omega)))
    Sexactm = np.concatenate((Sexact.real[0:h], Sexact.real[h+1:]))
    
    print('w shape', np.shape(w))
    print('Sexact shape', np.shape(Sexact))
    plt.plot(w, -Sexact)
    plt.plot(wm, -Sexactm/wm)
    
    plt.ylim(-3, 3)


    plt.savefig('data/sigma_and_deriv')    


# g is g n (b+bdag)
# lamb0 = 2 * g^2 * rho / omega = 2 * 1 * 0.5 / 1 = 1

#test_1d(100, 1.0)
#compute_rhoEF_1d()
#plot_1d()




'''
folder = '/scratch/users/bln/elph/data/single_iter/data/data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone/'

mu = np.load(folder + 'mu.npy')[0]


w = np.arange(params['wmin'], params['wmax'], params['dw'])

width = 0.05j
#S = single_iter.get_S(w, 1/6, params['beta'], params['omega'], params['nk'], params['idelta'], mu)
S = single_iter.get_S(w, 1/6, params['beta'], params['omega'], 120, width, mu)


path = os.path.join(folder, 'SR.npy')

SR = np.load(path)
print(SR.shape)

plt.figure()
plt.plot(w, S.real)
plt.plot(w, S.imag)

nr = len(w)
print('deriv exact sol', (S[nr//2+2] - S[nr//2-2])/(w[nr//2+2] - w[nr//2-2]))


plt.plot(w, SR[0,0].real)
plt.plot(w, SR[0,0].imag)

plt.savefig(basedir + 'S')


nr = len(w)
print('deriv migdal ', (SR[0,0,nr//2+2] - SR[0,0,nr//2-2])/(w[nr//2+2] - w[nr//2-2]))
'''
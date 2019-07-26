import matplotlib
matplotlib.use('TkAgg')
from matplotlib.pyplot import *
import src
import fourier
from convolution import conv
from scipy import stats
from numpy import *
from functions import band_square_lattice
from renormalized_2d import Migdal
import os

params = {}
params['nw']    = 512
params['nk']    = 12
params['t']     = 1.0
params['tp']    = -0.3 # -0.3
params['omega'] = 0.17 # 0.17
params['dens']  = 0.8
params['renormalized'] = True
params['sc']    = 1
params['band']  = band_square_lattice
params['beta']  = 16.0
params['g0']    = 0.125

set_printoptions(precision=3)

def linregress(x, y):
    return stats.linregress(x, y)[0]

def err_pi(plotting=False):
    beta = params['beta']
    N = params['nw']
    
    tau = linspace(0, beta, 2*N+1)
    wn = (2.0*arange(N)+1.0) * pi / beta
    vn = (2.0*arange(N+1)) * pi / beta
    
    E1 = 0.2
    E2 = 0.8
    Gw1 = 1.0/(1j*wn - E1)
    Gw2 = 1.0/(1j*wn - E2)

    nF1 = 1.0/(exp(beta*E1)+1.0)
    nF2 = 1.0/(exp(beta*E2)+1.0)
    exact = (nF1 - nF2)/(1j*vn + E1 - E2) 

    Gt1 = fourier.w2t(Gw1, beta, 0, 'fermion', jump=-1)
    Gt2 = fourier.w2t(Gw2, beta, 0, 'fermion', jump=-1) 
    prod = -Gt1[::-1] * Gt2
    Iw = fourier.t2w(prod, beta, 0, 'boson')    

    if plotting:
        figure()
        plot(exact.real)
        plot(Iw.real)
        xlim(0, 200/beta)
        title('real PI')
        
        figure()   
        plot(exact.imag)
        plot(Iw.imag)
        xlim(0, 200/beta)
        title('imag PI')
        show()

    return mean(abs(Iw-exact))

def test_pi():
    ntaus = [64, 128, 256, 512, 1024, 2048]
    beta = 10.0
    
    errs = []
    for ntau in ntaus:
        params['nw'] = ntau
        errs.append(err_pi())

    print('errors')
    print(errs)
        
    x, y = log10(ntaus), log10(errs)
    figure()
    plot(x, y)
    title('PI')
    ylabel('log(error)')
    xlabel('log(N)')
    show()

    print('order ', linregress(x, y))

def err_sigma(plotting=False):
    beta = params['beta']
    N    = params['nw']

    tau = linspace(0, beta, 2*N+1)
    wn = (2.0*arange(N)+1.0) * pi / beta
    vn = (2.0*arange(N+1)) * pi / beta
    
    E = 0.2
    omega = 0.5
    Gw0 = 1.0/(1j*wn - E)
    Dv0 = -2*omega/(vn**2 + omega**2)

    nF  = 1.0/(exp(beta*E)+1.0)
    nB  = 1.0/(exp(beta*omega)-1.0)
    exact = (nB + nF)/(1j*wn - E + omega) + (nB + 1 - nF)/(1j*wn - E - omega) 

    #Iw = -1.0 * conv(Gw0, Dv0, ['m,n-m'], [0], [False], beta, kinds=('fermion', 'boson', 'fermion'))
 
    Gt0  = fourier.w2t(Gw0, beta, 0, 'fermion', -1)
    Dt0  = fourier.w2t(Dv0, beta, 0, 'boson')
    prod = Gt0 * Dt0
    Iw   = -1.0 * fourier.t2w(prod, beta, 0, 'fermion')[0]       

    if plotting:
        figure()
        plot(exact.real)
        plot(Iw.real)
        xlim(0, 200/beta)
        title('real Sigma')

        figure()   
        plot(exact.imag)
        plot(Iw.imag)
        xlim(0, 200/beta)
        title('imag Sigma')
        show()

    return mean(abs(Iw-exact))

def test_sigma():
    ntaus = [64, 128, 256, 512, 1024, 2048]
    
    errs = []
    for ntau in ntaus:
        params['nw'] = ntau
        errs.append(err_sigma())

    print('errors')
    print(errs)
        
    x, y = log10(ntaus), log10(errs)
    figure()
    plot(x, y)
    title('sigma')
    ylabel('log(error)')
    xlabel('log(N)')
    show()

    print('order ', linregress(x, y))

    
def test_single_iteration():

    basedir = '/scratch/users/bln/elph/debug/'

    lamb = 0.6
    W    = 8.0
    params['g0'] = sqrt(0.5 * lamb / 2.4 * params['omega'] * W)
    params['nk'] = 2
    params['nw'] = 512
    params['beta'] = 16.0
    params['dens'] = 0.8
    params['omega'] = 0.5
    omega = params['omega']
    
    migdal = Migdal(params, basedir)
    S0, PI0, sc_iter  = None, None, 1
    savedir, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, frac=1.0)
    PI = params['g0']**2 * GG
    S = fourier.t2w(S, params['beta'], 2, 'fermion')[0]
    PI = fourier.t2w(PI, params['beta'], 2, 'boson')
    print('S from migdal')
    print(shape(S))

    savedir, wn, vn, ek, mu, deriv, dndmu = migdal.setup()

    print('E = ', params['band'](params['nk'], 1.0, params['tp']))    
    nk = params['nk']
    nw = params['nw']
    beta = params['beta']
    wn = (2.0*arange(nw)+1.0) * pi / beta
    vn = (2.0*arange(nw+1)) * pi / beta
    ekmu = params['band'](nk, 1.0, params['tp']) - mu

    Dv0 = -2.0*omega/(vn**2 + omega**2)
    nB  = 1.0/(exp(beta*omega)-1.0)
    
    S_ = zeros((nk,nk,nw), dtype=complex)
    for ik1 in range(nk):
        for ik2 in range(nk):
            for iq1 in range(nk):
                for iq2 in range(nk):

                    E = ekmu[iq1,iq2]
                    nF = 1.0/(exp(beta*E)+1.0)
                    
                    S_[ik1,ik2,:] += (nB + nF)/(1j*wn - E + omega) + (nB + 1 - nF)/(1j*wn - E - omega) 

    S_ *= params['g0']**2 / nk**2
    
    print('S-S_', mean(abs(S-S_)))

    figure()
    plot(ravel(S).imag-ravel(S_).imag)
    plot(ravel(S).real-ravel(S_).real)
    title('diff Skw')
    #savefig('figs/diff Skw.png')

    figure()
    plot(ravel(S).imag)
    plot(ravel(S_).imag)
    plot(ravel(S).real)
    plot(ravel(S_).real)
    title('Skw')
    #savefig('figs/Skw.png')    
    show()
    
    PI_ = zeros((nk,nk,nw+1), dtype=complex)
    for ik1 in range(nk):
        for ik2 in range(nk):
            for iq1 in range(nk):
                for iq2 in range(nk):
                    ip1 = ((ik1+iq1)-nk//2)%nk
                    ip2 = ((ik2+iq2)-nk//2)%nk
                    
                    E1 = ekmu[ik1,ik2]
                    E2 = ekmu[ip1,ip2]

                    nF1 = 1.0/(exp(beta*E1)+1.0)
                    nF2 = 1.0/(exp(beta*E2)+1.0)

                    if abs(E1-E2)<1e-14:
                        PI_[iq1,iq2,0] += -beta * nF1 * (1-nF1)     
                        PI_[iq1,iq2,1:] += (nF1 - nF2)/(1j*vn[1:] + E1 - E2)
                    else:
                        PI_[iq1,iq2,:] += (nF1 - nF2)/(1j*vn + E1 - E2)         

    PI_ *= 2.0 * params['g0']**2 / nk**2

    print('PI-PI_', mean(abs(PI-PI_)))

    '''
    figure()
    plot(ravel(PI).imag-ravel(PI_).imag)
    plot(ravel(PI).real-ravel(PI_).real)
    title('re diff PIkw')
    #savefig('figs/diff_PIkw.png')
    '''

    figure()
    plot(ravel(PI).imag)
    plot(ravel(PI_).imag)
    title('Im PIkw')

    figure()
    plot(ravel(PI).real)
    plot(ravel(PI_).real)
    title('Re PIkw')
    show()
        
def simple_test():
    beta = params['beta']
    nw = params['nw']

    tau = linspace(0, beta, 2*nw+1)
    wn = (2.0*arange(nw)+1.0) * pi / beta
    vn = (2.0*arange(nw+1)) * pi / beta
    
    E1 = 0.2
    E2 = 0.8
    Gw1 = 1.0/(1j*wn - E1)
    Gw2 = 1.0/(1j*wn - E2)

    nF1 = 1.0/(exp(beta*E1)+1.0)
    nF2 = 1.0/(exp(beta*E2)+1.0)
    exact = (nF1 - nF2)/(1j*vn + E1 - E2) 

    wr = random.randn(nw) + 1j*random.randn(nw)
    t1 = fourier.w2t(wr, beta, 0, 'fermion', -1)
    w1, jump = fourier.t2w(t1,  beta, 0, 'fermion')
    print('jump\n',jump)
    print('diff\n',mean(abs(w1 - wr)))
    figure()
    plot(wr.real-w1.real)
    show()

    t1 = fourier.w2t(wr, beta, 0, 'boson')
    w1 = fourier.t2w(t1, beta, 0, 'boson')
    print('diff\n',mean(abs(w1 - wr)))
    figure()
    plot(wr.real-w1.real)
    show()
    

if __name__=='__main__':

    #simple_test()
    #exit()

    #err_pi(plotting=True)
    
    #exit()
    #test_pi()
    
    #err_sigma(plotting=True)

    #exit()

    '''
    params['beta'] = 10.0

    err_sigma(fourier.w2t_fermion_alpha0, fourier.w2t_boson, fourier.t2w_fermion_alpha0, plotting=True)
    test_sigma()

    err_pi(fourier.w2t_fermion_alpha0, fourier.t2w_boson, plotting=True)
    test_pi()
    '''

    test_single_iteration()
    


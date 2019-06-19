import src
import fourier
from convolution import conv
from scipy import stats
from numpy import *
from matplotlib.pyplot import *
from params import params
from renormalized_2d import Migdal

set_printoptions(precision=3)

def linregress(x, y):
    return stats.linregress(x, y)[0]

def err_pi(w2t_method, t2w_method, plotting=False):
    beta = params['beta']
    N = params['Nw']
    
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

    Iw = conv(Gw1, Gw2, ['m,n+m'], [0], [False], beta, kinds=('fermion', 'fermion', 'boson'))
    
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
        params['Nw'] = ntau
        errs.append(err_pi(fourier.w2t_fermion_alpha0, fourier.t2w_boson))

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

def err_sigma(w2t_fermion_method, w2t_boson_method, t2w_fermion_method, plotting=False):
    beta = params['beta']
    N    = params['Nw']

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

    Iw = -1.0 * conv(Gw0, Dv0, ['m,n-m'], [0], [False], beta, kinds=('fermion', 'boson', 'fermion'))
        
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
        params['Nw'] = ntau
        errs.append(err_sigma(fourier.w2t_fermion_alpha0, fourier.w2t_boson, fourier.t2w_fermion_alpha0))

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

    lamb = 0.6
    W    = 8.0
    params['g0'] = sqrt(0.5 * lamb / 2.4 * params['omega'] * W)
    params['Nk'] = 2
    params['Nw'] = 512
    params['beta'] = 1.0
    omega = params['omega']
    
    migdal = Migdal(params)
    S0, PI0, sc_iter  = None, None, 1
    savedir, G, D, S, PI = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, frac=1.0)

    print('E = ', params['band'](params['Nk']))    

    Nk = params['Nk']
    Nw = params['Nw']
    beta = params['beta']
    wn = (2.0*arange(Nw)+1.0) * pi / beta
    vn = (2.0*arange(Nw+1)) * pi / beta
    ek = params['band'](Nk)

    Dv0 = -2.0*omega/(vn**2 + omega**2)
    nB  = 1.0/(exp(beta*omega)-1.0)
    
    S_ = zeros((Nk,Nk,Nw), dtype=complex)
    for ik1 in range(Nk):
        for ik2 in range(Nk):
            for iq1 in range(Nk):
                for iq2 in range(Nk):

                    E = ek[iq1,iq2]
                    nF = 1.0/(exp(beta*E)+1.0)
                    
                    S_[ik1,ik2,:] += (nB + nF)/(1j*wn - E + omega) + (nB + 1 - nF)/(1j*wn - E - omega) 

    S_ *= params['g0']**2 / Nk**2
    
    print('S-S_', mean(abs(S-S_)))

    figure()
    plot(ravel(S)-ravel(S_))
    title('diff Skw')
    
    figure()
    plot(ravel(S))
    plot(ravel(S_))
    title('Skw')

    show()
    
    PI_ = zeros((Nk,Nk,Nw+1), dtype=complex)
    for ik1 in range(Nk):
        for ik2 in range(Nk):
            for iq1 in range(Nk):
                for iq2 in range(Nk):
                    ip1 = ((ik1+iq1)-Nk//2)%Nk
                    ip2 = ((ik2+iq2)-Nk//2)%Nk
                    
                    E1 = ek[ik1,ik2]
                    E2 = ek[ip1,ip2]

                    nF1 = 1.0/(exp(beta*E1)+1.0)
                    nF2 = 1.0/(exp(beta*E2)+1.0)

                    if abs(E1-E2)<1e-14:
                        PI_[iq1,iq2,0] += -beta * nF1 * (1-nF1)     
                        PI_[iq1,iq2,1:] += (nF1 - nF2)/(1j*vn[1:] + E1 - E2)
                    else:
                        PI_[iq1,iq2,:] += (nF1 - nF2)/(1j*vn + E1 - E2)         

    PI_ *= 2.0 * params['g0']**2 / Nk**2

    print('PI-PI_', mean(abs(PI-PI_)))

    figure()
    plot(ravel(PI)-ravel(PI_))
    title('diff PIkw')
    
    figure()
    plot(ravel(PI))
    plot(ravel(PI_))
    title('PIkw')
    show()
        

if __name__=='__main__':

    '''
    params['beta'] = 10.0

    err_sigma(fourier.w2t_fermion_alpha0, fourier.w2t_boson, fourier.t2w_fermion_alpha0, plotting=True)
    test_sigma()

    err_pi(fourier.w2t_fermion_alpha0, fourier.t2w_boson, plotting=True)
    test_pi()
    '''

    test_single_iteration()
    


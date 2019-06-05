import src
import fourier
from convolution import conv
from scipy import stats
from numpy import *
from matplotlib.pyplot import *
from params import params

def linregress(x, y):
    return stats.linregress(x, y)[0]

def err_pi(w2t_method, t2w_method, plotting=False):
    beta = params.beta
    N = params.Nw
    
    tau = np.linspace(0, beta, 2*N+1)
    wn = (2.0*np.arange(N)+1.0) * np.pi / beta
    vn = (2.0*np.arange(N+1)) * np.pi / beta
    
    E1 = 0.2
    E2 = 0.8
    Gw1 = 1.0/(1j*wn - E1)
    Gw2 = 1.0/(1j*wn - E2)

    nF1 = 1.0/(np.exp(beta*E1)+1.0)
    nF2 = 1.0/(np.exp(beta*E2)+1.0)
    exact = (nF1 - nF2)/(1j*vn + E1 - E2) 

    Iw = conv(Gw1, Gw2, ['m,n+m'], [0], [False], params, kinds=('fermion', 'fermion', 'boson'))
    
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

    return np.mean(np.abs(Iw-exact))

def test_pi():
    ntaus = [64, 128, 256, 512, 1024, 2048]
    beta = 10.0
    
    errs = []
    for ntau in ntaus:
        params.Nw = ntau
        errs.append(err_pi(fourier.w2t_fermion_alpha0, fourier.t2w_boson))

    print('errors')
    print(errs)
        
    x, y = np.log10(ntaus), np.log10(errs)
    figure()
    plot(x, y)
    show()

    print('order ', linregress(x, y))

def err_sigma(w2t_fermion_method, w2t_boson_method, t2w_fermion_method, plotting=False):
    beta = params.beta
    N    = params.Nw

    tau = np.linspace(0, beta, 2*N+1)
    wn = (2.0*np.arange(N)+1.0) * np.pi / beta
    vn = (2.0*np.arange(N+1)) * np.pi / beta
    
    E = 0.2
    omega = 0.5
    Gw0 = 1.0/(1j*wn - E)
    Dv0 = -2*omega/(vn**2 + omega**2)

    nF  = 1.0/(np.exp(beta*E)+1.0)
    nB  = 1.0/(np.exp(beta*omega)-1.0)
    exact = (nB + nF)/(1j*wn - E + omega) + (nB + 1 - nF)/(1j*wn - E - omega) 

    Iw = -1.0 * conv(Gw0, Dv0, ['m,n-m'], [0], [False], params, kinds=('fermion', 'boson', 'fermion'))
        
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

    return np.mean(np.abs(Iw-exact))

def test_sigma():
    ntaus = [64, 128, 256, 512, 1024, 2048]
    
    errs = []
    for ntau in ntaus:
        params.Nw = ntau
        errs.append(err_sigma(fourier.w2t_fermion_alpha0, fourier.w2t_boson, fourier.t2w_fermion_alpha0))

    print('errors')
    print(errs)
        
    x, y = np.log10(ntaus), np.log10(errs)
    figure()
    plot(x, y)
    show()

    print('order ', linregress(x, y))

params.beta = 10.0
    
err_sigma(fourier.w2t_fermion_alpha0, fourier.w2t_boson, fourier.t2w_fermion_alpha0, plotting=True)
test_sigma()

err_pi(fourier.w2t_fermion_alpha0, fourier.t2w_boson, plotting=True)
test_pi()



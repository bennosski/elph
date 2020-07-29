import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, iv
import src
import fourier
from scipy import optimize


def lorentzian(w, x, width):
    return 1/np.pi * width / ((w-x)**2 + width**2)


def one_site_zero_T(w, g, ec, lmax, width):
    omega = 1.0
    delta = g*omega
    
    A = 0
    for il in range(lmax):
        A += g**il / factorial(il) * lorentzian(w, ec - delta + omega*il, width)
    A *= np.exp(-g)
    
    return A
    

def one_site_G(w, g, ec, lmax, width, beta):
    omega = 1.0
    delta = g*omega
    nB = 1/(np.exp(omega*beta) - 1)
    
    arg = 2*g * np.sqrt(nB*(nB+1))
    G = 0
    for il in range(-lmax, lmax):
        G += iv(il, arg) * np.exp(il*beta*omega/2) / (w - ec + delta - omega*il + 1j*width)
    G *= np.exp(-g*(2*nB+1))

    return G


def one_site_G_matsubara(iwn, g, ec, lmax, beta):
    omega = 1.0
    delta = g*omega
    nB = 1/(np.exp(omega*beta) - 1)
    
    arg = 2*g * np.sqrt(nB*(nB+1))
    G = 0
    for il in range(-lmax, lmax):
        G += iv(il, arg) * np.exp(il*beta*omega/2) / (iwn - ec + delta - omega*il)
    G *= np.exp(-g*(2*nB+1))

    return G
    
    

def S_from_G(G, w, ec, width):
    return w - ec + 1j*width - 1/G



def one_site_D(w, g, ec, lmax, width):
    pass



def weak_coupling():
    
    w = np.linspace(-4, 4, 20000)
    width = 0.001
    #width = 0.1
    #g, ec = 0.1, 0.1
    lmax = 10
    beta = 20
    #g, ec = 0.05, 0.05
    g, ec = 5.5, 0
    omega = 1
    
    
    A = one_site_zero_T(w, g, ec, lmax, width)
    plt.figure()
    plt.plot(w, A)
    print('norm', np.trapz(A, dx=(w[-1]-w[0])/(len(w)-1)))
    
    '''
    g = 5.5
    
    w = np.linspace(-10, 10, 1000)
    A = one_site_zero_T(w, g, ec, lmax, width)
    plt.figure()
    plt.plot(w, A)
    plt.xlim(-5, 5)
    '''
    
    
    G = one_site_G(w, g, ec, lmax, width, beta)
    
    A = -1/np.pi * G.imag
    plt.figure()
    plt.plot(w, A)
    plt.xlim(-2.5, 2.5)
    plt.ylabel('$A(\omega)$', fontsize=14)
    plt.xlabel('$\omega$', fontsize=14)
    plt.tight_layout()
    plt.savefig('data/Aexact')
    
    
    nF = 1/(np.exp(w*beta) + 1)
    dw = (w[-1]-w[0])/(len(w)-1)
    print('norm', np.trapz(A, dx=dw))
    print('filling from A(k,w) : ', 2*np.trapz(A*nF, dx=dw))
    
    S = S_from_G(G, w, ec, width)
    plt.figure()
    plt.plot(w, S.real)
    plt.plot(w, S.imag)
    plt.xlim(-1, 5)
    plt.title('S')
    
    
    nw = 2048
    iwn = np.pi*(2*np.arange(nw)+1)*1j/beta
    
    G_matsubara = one_site_G_matsubara(iwn, g, ec, lmax, beta)
    
    n = (1.0 + 2./beta * 2.0*sum(G_matsubara)).real
    baresum = 1 + np.tanh(-beta*ec/2)
    bareGw = 1.0/(iwn - ec)
    ntail = baresum - (1 + 2/beta * 2*np.sum(bareGw.real))
    
    print('filling from Gw ', n + ntail)
    
    Gtau = fourier.w2t(G_matsubara, beta, axis=0, kind='fermion', jump=-1)
    
    print('filling from Gtau ', -2*Gtau[-1].real)
    
    
    tau = np.linspace(0, beta, 200)[:,None]
    w_ = w[None,:]
    K =  np.exp(-tau*w_) / (1 + np.exp(-beta*w_))   # plus sign?
    Gtau_from_Akw = - np.dot(K, A) * dw
    
    plt.figure()
    plt.plot(np.linspace(0, beta, len(Gtau)), Gtau.real)
    plt.plot(np.linspace(0, beta, len(Gtau_from_Akw)), Gtau_from_Akw.real)
    plt.title('Gtaus') 
    plt.ylim(-1, 0)
    
    print('filling from Gtau_from_Akw', -2*Gtau_from_Akw[-1].real)
    
    
    #Gtau_ume = np.load('/scratch/users/bln/elph/data/onesite/data/data_unrenormalized_nk0_abstp0.000_dim0_g00.31623_nw2048_omega1.000_dens0.905_beta20.0000_QNone/G.npy')
    #Gtau_rme = np.load('/scratch/users/bln/elph/data/onesite/data/data_renormalized_nk0_abstp0.000_dim0_g00.31623_nw2048_omega1.000_dens0.905_beta20.0000_QNone/G.npy')
    
    Gtau_ume = np.load('/scratch/users/bln/elph/data/onesite/data/data_unrenormalized_nk0_abstp0.000_dim0_g00.22361_nw2048_omega1.000_dens0.951_beta20.0000_QNone/G.npy')
    Gtau_rme = np.load('/scratch/users/bln/elph/data/onesite/data/data_renormalized_nk0_abstp0.000_dim0_g00.22361_nw2048_omega1.000_dens0.951_beta20.0000_QNone/G.npy')
    
    
    # -----------------------------------------------------------------------------
    # G single iter:
    def get_Gw_si(mu):
        Ec = ec - mu
        nB = 1/(np.exp(beta*omega) - 1)
        nF = 1/(np.exp(beta*Ec) + 1)
        S = 1.0 / beta * ((nB + 1 - nF)/(iwn - Ec - omega) + (nB + nF)/(iwn - Ec + omega))
        return 1/(iwn - Ec - S)
        
    def get_Gtau_si(mu):
        Gw = get_Gw_si(mu)
        return fourier.w2t(Gw, beta, axis=0, kind='fermion', jump=-1)
    
    def get_fill(mu):
        Gw = get_Gw_si(mu)
        Ec = ec - mu
    
        n = (1.0 + 2./beta * 2.0*sum(Gw)).real
    
        baresum = 1 + np.tanh(-beta*Ec/2)
        bareGw = 1.0/(iwn - Ec)
        ntail = baresum - (1 + 2/beta * 2*np.sum(bareGw.real))
        
        return n + ntail
    
    #mu = 0
    mu_new = optimize.fsolve(lambda mu : get_fill(mu) - 0.951, 0)[0]
    Gtau_si = get_Gtau_si(mu_new)
    Gw_si = get_Gw_si(mu_new)
    print('filling si', -2*Gtau_si[-1].real)
    #----------------------------------------------------------
    
    S_matsubara = iwn - ec - 1/G_matsubara
    S_si = iwn - (ec - mu_new)  - 1/Gw_si
    S_rme = np.load('/scratch/users/bln/elph/data/onesite/data/data_renormalized_nk0_abstp0.000_dim0_g00.22361_nw2048_omega1.000_dens0.951_beta20.0000_QNone/S.npy')
    
    plt.figure()
    plt.plot(iwn.imag, G_matsubara.real)
    plt.plot(iwn.imag, Gw_si.real)
    plt.xlim(0, 3)
    plt.title('Gw')
    
    plt.figure()
    plt.plot(iwn.imag, S_matsubara.real)
    plt.plot(iwn.imag, S_si.real)
    wn_rme = (2*np.arange(len(S_rme))+1)*np.pi/beta
    plt.plot(wn_rme, S_rme.real)
    plt.xlim(0, 3)
    plt.title('Sw')
    
    
    f = plt.figure()
    ax = f.add_axes([0.16, 0.13, 0.82, 0.82])
    ax.plot(np.linspace(0, beta, len(Gtau)), Gtau.real)
    ax.plot(np.linspace(0, beta, len(Gtau_ume)), Gtau_ume.real)
    ax.plot(np.linspace(0, beta, len(Gtau_rme)), Gtau_rme.real)
    ax.legend(['exact', 'unrenormalized ME', 'renormalized ME'])
    ax.set_ylabel(r'$G(\tau)$', fontsize=14)
    ax.set_xlabel(r'$\tau$', fontsize=14)
    plt.savefig('data/Gtau')
    
    
    A_ume = np.load('/scratch/users/bln/elph/data/onesite/data/data_unrenormalized_nk0_abstp0.000_dim0_g00.22361_nw2048_omega1.000_dens0.951_beta20.0000_QNone/GR.npy')
    A_rme = np.load('/scratch/users/bln/elph/data/onesite/data/data_renormalized_nk0_abstp0.000_dim0_g00.22361_nw2048_omega1.000_dens0.951_beta20.0000_QNone/GR.npy')
    ws = np.load('/scratch/users/bln/elph/data/onesite/data/data_unrenormalized_nk0_abstp0.000_dim0_g00.22361_nw2048_omega1.000_dens0.951_beta20.0000_QNone/w.npy')
    
    A_ume = -1/np.pi * A_ume.imag
    A_rme = -1/np.pi * A_rme.imag
    
    plt.figure()
    plt.plot(w, A)
    plt.plot(ws, A_ume, '--')
    plt.plot(ws, A_rme*4)
    plt.legend(['$A_{exact}$', '$A_{UME}$', r'$4 \times A_{RME}$'], loc=2)
    plt.ylabel('$A(\omega)$', fontsize=14)
    plt.xlabel('$\omega$', fontsize=14)
    plt.ylim(0, 20)
    plt.xlim(-2.1, 2.1)
    plt.savefig('data/A')



def strong_coupling():
    
    w = np.linspace(-10, 10, 60000)
    width = 0.001
    #width = 0.1
    #g, ec = 0.1, 0.1
    lmax = 30
    beta = 20
    #g, ec = 0.05, 0.05
    g, ec = 5.5, 0
    omega = 1
    
    
    A = one_site_zero_T(w, g, ec, lmax, width)
    plt.figure()
    plt.plot(w, A)
    print('norm', np.trapz(A, dx=(w[-1]-w[0])/(len(w)-1)))
   
    
    G = one_site_G(w, g, ec, lmax, width, beta)
    
    A = -1/np.pi * G.imag
    plt.figure()
    plt.plot(w, A)
    plt.xlim(-7, 7)
    plt.ylabel('$A(\omega)$', fontsize=14)
    plt.xlabel('$\omega$', fontsize=14)
    plt.tight_layout()
    plt.savefig('data/Aexact')
    
    
    nF = 1/(np.exp(w*beta) + 1)
    dw = (w[-1]-w[0])/(len(w)-1)
    print('norm', np.trapz(A, dx=dw))
    print('filling from A(k,w) : ', 2*np.trapz(A*nF, dx=dw))
    
    S = S_from_G(G, w, ec, width)
    plt.figure()
    plt.plot(w, S.real)
    plt.plot(w, S.imag)
    plt.xlim(-1, 5)
    plt.title('S')
    
    
    nw = 2048
    iwn = np.pi*(2*np.arange(nw)+1)*1j/beta
    
    G_matsubara = one_site_G_matsubara(iwn, g, ec, lmax, beta)
    
    n = (1.0 + 2./beta * 2.0*sum(G_matsubara)).real
    baresum = 1 + np.tanh(-beta*ec/2)
    bareGw = 1.0/(iwn - ec)
    ntail = baresum - (1 + 2/beta * 2*np.sum(bareGw.real))
    
    print('filling from Gw ', n + ntail)
    
    Gtau = fourier.w2t(G_matsubara, beta, axis=0, kind='fermion', jump=-1)
    
    print('filling from Gtau ', -2*Gtau[-1].real)
    
    
    tau = np.linspace(0, beta, 200)[:,None]
    w_ = w[None,:]
    K =  np.exp(-tau*w_) / (1 + np.exp(-beta*w_))   # plus sign?
    Gtau_from_Akw = - np.dot(K, A) * dw
    
    plt.figure()
    plt.plot(np.linspace(0, beta, len(Gtau)), Gtau.real)
    plt.plot(np.linspace(0, beta, len(Gtau_from_Akw)), Gtau_from_Akw.real)
    plt.title('Gtaus') 
    plt.ylim(-1, 0)
    
    print('filling from Gtau_from_Akw', -2*Gtau_from_Akw[-1].real)
    
    
    #Gtau_ume = np.load('/scratch/users/bln/elph/data/onesite/data/data_unrenormalized_nk0_abstp0.000_dim0_g00.31623_nw2048_omega1.000_dens0.905_beta20.0000_QNone/G.npy')
    #Gtau_rme = np.load('/scratch/users/bln/elph/data/onesite/data/data_renormalized_nk0_abstp0.000_dim0_g00.31623_nw2048_omega1.000_dens0.905_beta20.0000_QNone/G.npy')
    
    Gtau_ume = np.load('/scratch/users/bln/elph/data/onesite/data/data_unrenormalized_nk0_abstp0.000_dim0_g02.34521_nw2048_omega1.000_dens1.058_beta20.0000_QNone/G.npy')
    #Gtau_rme = np.load('/scratch/users/bln/elph/data/onesite/data/data_renormalized_nk0_abstp0.000_dim0_g00.22361_nw2048_omega1.000_dens0.951_beta20.0000_QNone/G.npy')
    
    
    # -----------------------------------------------------------------------------
    # G single iter:
    def get_Gw_si(mu):
        Ec = ec - mu
        nB = 1/(np.exp(beta*omega) - 1)
        nF = 1/(np.exp(beta*Ec) + 1)
        S = 1.0 / beta * ((nB + 1 - nF)/(iwn - Ec - omega) + (nB + nF)/(iwn - Ec + omega))
        return 1/(iwn - Ec - S)
        
    def get_Gtau_si(mu):
        Gw = get_Gw_si(mu)
        return fourier.w2t(Gw, beta, axis=0, kind='fermion', jump=-1)
    
    def get_fill(mu):
        Gw = get_Gw_si(mu)
        Ec = ec - mu
    
        n = (1.0 + 2./beta * 2.0*sum(Gw)).real
    
        baresum = 1 + np.tanh(-beta*Ec/2)
        bareGw = 1.0/(iwn - Ec)
        ntail = baresum - (1 + 2/beta * 2*np.sum(bareGw.real))
        
        return n + ntail
    
    #mu = 0
    mu_new = optimize.fsolve(lambda mu : get_fill(mu) - 0.951, 0)[0]
    Gtau_si = get_Gtau_si(mu_new)
    Gw_si = get_Gw_si(mu_new)
    print('filling si', -2*Gtau_si[-1].real)
    #----------------------------------------------------------
    
    S_matsubara = iwn - ec - 1/G_matsubara
    S_si = iwn - (ec - mu_new)  - 1/Gw_si
    S_rme = np.load('/scratch/users/bln/elph/data/onesite/data/data_renormalized_nk0_abstp0.000_dim0_g00.22361_nw2048_omega1.000_dens0.951_beta20.0000_QNone/S.npy')
    
    plt.figure()
    plt.plot(iwn.imag, G_matsubara.real)
    plt.plot(iwn.imag, Gw_si.real)
    plt.xlim(0, 3)
    plt.title('Gw')
    
    plt.figure()
    plt.plot(iwn.imag, S_matsubara.real)
    plt.plot(iwn.imag, S_si.real)
    wn_rme = (2*np.arange(len(S_rme))+1)*np.pi/beta
    plt.plot(wn_rme, S_rme.real)
    plt.xlim(0, 3)
    plt.title('Sw')
    
    
    f = plt.figure()
    ax = f.add_axes([0.16, 0.13, 0.82, 0.82])
    ax.plot(np.linspace(0, beta, len(Gtau)), Gtau.real)
    ax.plot(np.linspace(0, beta, len(Gtau_ume)), Gtau_ume.real)
    #ax.plot(np.linspace(0, beta, len(Gtau_rme)), Gtau_rme.real)
    ax.legend(['exact', 'unrenormalized ME', 'renormalized ME'])
    ax.set_ylabel(r'$G(\tau)$', fontsize=14)
    ax.set_xlabel(r'$\tau$', fontsize=14)
    plt.savefig('data/Gtau')
    
    ws = np.load('/scratch/users/bln/elph/data/onesite/data/data_unrenormalized_nk0_abstp0.000_dim0_g02.34521_nw2048_omega1.000_dens1.058_beta20.0000_QNone/w.npy')
    A_ume = np.load('/scratch/users/bln/elph/data/onesite/data/data_unrenormalized_nk0_abstp0.000_dim0_g02.34521_nw2048_omega1.000_dens1.058_beta20.0000_QNone/GR.npy')
    #A_rme = np.load('/scratch/users/bln/elph/data/onesite/data/data_renormalized_nk0_abstp0.000_dim0_g00.22361_nw2048_omega1.000_dens0.951_beta20.0000_QNone/GR.npy')
    #ws = np.load('/scratch/users/bln/elph/data/onesite/data/data_unrenormalized_nk0_abstp0.000_dim0_g00.22361_nw2048_omega1.000_dens0.951_beta20.0000_QNone/w.npy')
    
    A_ume = -1/np.pi * A_ume.imag
    #A_rme = -1/np.pi * A_rme.imag
    
    plt.figure()
    plt.plot(w, A)
    plt.plot(ws, A_ume, '--')
    #plt.plot(ws, A_rme*4)
    plt.legend(['$A_{exact}$', '$A_{UME}$', r'$4 \times A_{RME}$'], loc=2)
    plt.ylabel('$A(\omega)$', fontsize=14)
    plt.xlabel('$\omega$', fontsize=14)
    plt.ylim(0, 40)
    plt.xlim(-7, 10)
    plt.savefig('data/A')


strong_coupling()


def test():
    zs = np.linspace(-1, 1, 100)
    thetas = np.linspace(-np.pi, np.pi, 100)
    rhs = np.exp(zs[:,None] * np.cos(thetas)[None,:])
    lhs = 0
    for il in range(-10, 10):
        lhs += iv(il, zs)[:,None] * np.exp(1j*il*thetas)[None,:]
    
    plt.figure()
    plt.imshow(rhs)
    plt.colorbar()
    
    plt.figure()
    plt.imshow(lhs.real)
    plt.colorbar()
    
    
    
    
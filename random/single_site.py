import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, iv


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


def S_from_G(G, w, ec, width):
    return w - ec + 1j*width - 1/G



def one_site_D(w, g, ec, lmax, width):
    pass



w = np.linspace(-5, 5, 1000)
width = 0.05
ec = 0

A = one_site_zero_T(w, 0.5, ec, 10, width)
plt.figure()
plt.plot(w, A)
print('norm', np.trapz(A, dx=(w[-1]-w[0])/(len(w)-1)))

w = np.linspace(-10, 10, 1000)
A = one_site_zero_T(w, 5.5, ec, 20, width)
plt.figure()
plt.plot(w, A)
plt.xlim(-5, 5)


G = one_site_G(w, 0.5, ec, 10, width, 20)

A = -1/np.pi * G.imag
plt.figure()
plt.plot(w, A)
plt.xlim(-5, 5)
plt.title('A high T')

print('norm', np.trapz(A, dx=(w[-1]-w[0])/(len(w)-1)))

S = S_from_G(G, w, ec, width)
plt.figure()
plt.plot(w, S.real)
plt.plot(w, S.imag)
plt.xlim(-1, 5)
plt.title('S')




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
    
    
    
    
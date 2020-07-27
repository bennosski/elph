
import imagaxis
import numpy as np
from functions import band_square_lattice, mylamb2g0
from scipy import optimize

def compute_fill(ek, beta, mu, nk):
    nF = 1/(np.exp(beta*(ek-mu)) + 1)
    return 2 * np.sum(nF) / nk**2
    

def compute_mu(ek, beta, nk, dens):
    mu = optimize.fsolve(lambda mu : compute_fill(ek, beta, mu, nk) - dens, 0)
    print('mu = ', mu)
    return mu
    

def get_S(w, ek, g0, beta, omega, nk, idelta, dens, mu):
    print('g0 = ', g0)

    #ek = band_square_lattice(nk, 1, -0.3)
    ek = ek[:,:,None] - mu

    nB = 1/(np.exp(beta*omega) - 1)
    nF = 1/(np.exp(beta*ek) + 1)
    w = w[None,None,:]

    S = g0**2 / nk**2 * np.sum((nB + nF) / (w + omega - ek + idelta) + (nB + 1 - nF) / (w - omega - ek + idelta), axis=(0,1))
    S = S - S[len(S)//2].real
    return mu, S


def get_S_constant_N(w, ek, g0, beta, omega, nk, idelta, dens, mu):
    ek = ek[:,None] - mu

    nB = 1/(np.exp(beta*omega) - 1)
    nF = 1/(np.exp(beta*ek) + 1)
    w = w[None,:]

    S = g0**2 / nk * np.sum((nB + nF) / (w + omega - ek + idelta) + (nB + 1 - nF) / (w - omega - ek + idelta), axis=(0))
    S = S - S[len(S)//2].real
    return mu, S


import imagaxis
import numpy as np
from functions import band_square_lattice, mylamb2g0


def get_S(w, lamb, beta, omega, nk, idelta, mu):
    g0 = mylamb2g0(lamb, omega, 8)
    print('g0 = ', g0)

    ek = band_square_lattice(nk, 1, -0.3)
    ek = ek[:,:,None] - mu


    nB = 1/(np.exp(beta*omega)-1)
    nF = 1/(np.exp(beta*ek)+1)
    w = w[None,None,:]

    S = g0**2 / nk**2 * np.sum((nB + nF) / (w + omega - ek + idelta) + (nB + 1 - nF) / (w - omega - ek + idelta), axis=(0,1))

    return S

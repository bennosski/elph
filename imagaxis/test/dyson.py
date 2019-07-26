import matplotlib
matplotlib.use('TkAgg')
from matplotlib.pyplot import *
import src
import fourier
from convolution import conv
from scipy import stats
import numpy as np
from functions import band_square_lattice
import os

np.set_printoptions(precision=3)

def linregress(x, y):
    return stats.linregress(x, y)[0]

def dyson(wn, beta, H, St, axis):
    dim = np.shape(H)[0]
    Sw, jumpS = fourier.t2w(St, beta, axis, 'fermion')
    tau0 = np.identity(dim)
    Gw = np.linalg.inv(1j*wn[:,None,None]*tau0 - H[None,:,:] - Sw)
    jumpG = -np.identity(dim)[None,:,:]
    return fourier.w2t(Gw, beta, axis, 'fermion', jumpG)
  
def test_dyson():
    beta = 16.0
    nw   = 128
    dim  = 1
    norb = 3

    wn =(2*np.arange(nw)+1) * np.pi / beta
    H = np.random.randn(norb, norb)
    H += np.transpose(H)

    # solve full dyson for bare Green's function
    St = np.zeros((2*nw+1, norb, norb))
    Gt_exact = dyson(wn, beta, H, St, 0)

    figure()
    plot(Gt_exact[:,0,0].real)
    plot(Gt_exact[:,0,0].imag)
    title('exact G')

    # solve for 11 component using embedding selfenergy
    St = np.zeros((2*nw+1, norb-dim, norb-dim))
    Gt_bath = dyson(wn, beta, H[dim:,dim:], St, 0)

    St = np.einsum('ab,wbc,cd->wad', H[:dim,dim:], Gt_bath,  H[dim:,:dim])
    Gt = dyson(wn, beta, H[:dim, :dim], St, 0)

    print('diff')
    print(np.mean(np.abs(Gt-Gt_exact[:,:dim,:dim])))

    figure()
    plot(Gt[:,0,0].real)
    plot(Gt[:,0,0].imag)
    title('embedding test')
    show()

def old_test_dyson():
    '''
    embedding selfenergy test
    '''
    
    beta = 16.0
    nw   = 128
    dim  = 1
    norb = 3

    wn =(2*np.arange(nw)+1) * np.pi / beta

    H = np.random.randn(norb, norb)
    H += np.transpose(H)

    # solve full dyson for bare Green's function

    tau0   = np.identity(norb)
    Gw_exact = np.linalg.inv(1j*wn[:,None,None]*tau0 - H[None,:,:])
    jump = -np.identity(norb)
    Gt_exact = fourier.w2t(Gw_exact, beta, 0, 'fermion', jump)  

    figure()
    plot(Gt_exact[:,0,0].real)
    plot(Gt_exact[:,0,0].imag)
    title('exact G')

    # solve for 11 component using embedding selfenergy
  
    tau0  = np.identity(norb-dim)
    Gw_bath = np.linalg.inv(1j*wn[:,None,None]*tau0 - H[None,dim:,dim:]) 
    jump = -np.identity(norb-dim)
    Gt_bath = fourier.w2t(Gw_bath, beta, 0, 'fermion', jump)

    St = np.einsum('ab,wbc,cd->wad', H[:dim,dim:], Gt_bath,  H[dim:,:dim])
    Sw = fourier.t2w(St, beta, 0, 'fermion')[0]

    tau0 = np.identity(dim)
    Gw = np.linalg.inv(1j*wn[:,None,None]*tau0 - H[None,:dim,:dim] - Sw)
    jump = -np.identity(dim)
    Gt = fourier.w2t(Gw, beta, 0, 'fermion', jump)

    print('diff')
    print(np.mean(np.abs(Gt-Gt_exact[:,:dim,:dim])))

    figure()
    plot(Gt[:,0,0].real)
    plot(Gt[:,0,0].imag)
    title('embedding test')
    show()

if __name__=='__main__':
    test_dyson()


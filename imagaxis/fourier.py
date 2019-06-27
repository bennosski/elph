import numpy as np
from matplotlib.pyplot import *

def W_fermion(t):
    return 2.0*(1.0-np.cos(t))/t**2

def W_boson(t):
    out = np.zeros(len(t), dtype=complex)
    out[0] = 1.0
    out[1:] = 2.0*(1.0-np.cos(t[1:]))/t[1:]**2
    return out
    
def alpha0(t):
    return -(1.0-np.cos(t))/t**2 + 1j*(t-np.sin(t))/t**2

def w2t(x, beta, axis, kind=None):
    if kind=='fermion':
        return w2t_fermion_alpha0(x, beta, axis)
    elif kind=='boson':
        return w2t_boson(x, beta, axis)

def t2w(x, beta, axis, kind=None):
    if kind=='fermion':
        return t2w_fermion_alpha0(x, beta, axis)
    elif kind=='boson':
        return t2w_boson(x, beta, axis)

def expand(arr, axis, dim):
    shape = np.ones(dim, dtype=np.int)
    shape[axis] = len(arr)
    return np.reshape(arr, shape)

def t2w_fermion_alpha0(h, beta, axis):
    hshape = np.shape(h)
    dim = len(hshape)

    N = hshape[axis]-1
    delta = beta/N
    w  = np.fft.fftfreq(2*N, beta/(np.pi*2*N))[1::2]
    theta = w*delta

    indices = expand(np.arange(N), axis, dim)
    h_ = np.concatenate((np.take_along_axis(h, indices, axis), -np.take_along_axis(h, indices, axis)), axis=axis)

    jump = np.take_along_axis(h, expand(np.arange(1), axis, dim), axis) + np.take_along_axis(h, expand(np.arange(N,N+1), axis, dim), axis)
    I = expand(W_fermion(theta), axis, dim) * 0.5 * np.take_along_axis(np.fft.ifft(h_, axis=axis), expand(np.arange(1,2*N,2), axis, dim), axis)*2*N + expand(alpha0(theta), axis, dim)*jump
    I *= delta
    return np.take_along_axis(I, expand(np.arange(N//2), axis, dim), axis=axis)

'''
def t2w_fermion_alpha0_old(h, beta):
    N     = len(h)-1
    delta = beta/N
    w  = np.fft.fftfreq(2*N, beta/(np.pi*2*N))[1::2]
    theta = w*delta
    h_ = np.concatenate((h[:-1], -h[:-1]))
    I = W_fermion(theta) * 0.5 * np.fft.ifft(h_)[1::2]*2*N + alpha0(theta)*(h[0]+h[-1])
    I *= delta
    return I[:N//2]
''' 

def w2t_fermion_alpha0(I, beta, axis):
    # assume jump is -1

    Ishape = np.shape(I)
    dim = len(Ishape)

    N     = 2*Ishape[axis]
    delta = beta/N
    w  = np.fft.fftfreq(2*N, beta/(np.pi*2*N))[1::2]
    theta = w*delta

    # extend to negative frequencies
    I = np.concatenate((I, np.take_along_axis(np.conj(I), expand(np.arange(N//2-1,-1,-1), axis, dim), axis)), axis)
    
    I = 2.0*(I - expand(alpha0(theta), axis, dim)*(-1)*delta)/(delta * expand(W_fermion(theta), axis, dim))
    
    shape = np.array(np.shape(I))
    shape[axis] = 2*N

    I_ = np.zeros(shape, dtype=complex)
    np.put_along_axis(I_, expand(np.arange(1, 2*N, 2), axis, dim), I, axis)

    shape = np.array(np.shape(I))
    shape[axis] = N + 1
    out = np.zeros(shape, dtype=complex)
    np.put_along_axis(out, expand(np.arange(N), axis, dim), np.take_along_axis(np.fft.fft(I_, axis=axis), expand(np.arange(N), axis, dim), axis)/(2*N), axis)

    np.put_along_axis(out, expand(np.arange(N, N+1), axis, dim), -np.take_along_axis(out, expand(np.arange(1), axis, dim), axis) - 1.0, axis)
    return out

    
    '''
def w2t_fermion_alpha0_old(I, beta):
    N     = 2*len(I)
    delta = beta/N
    w  = np.fft.fftfreq(2*N, beta/(np.pi*2*N))[1::2]
    theta = w*delta

    # extend to negative frequencies
    I = np.concatenate((I, np.conj(I)[::-1]))
    
    I = 2.0*(I - alpha0(theta)*(-1)*delta)/(delta * W_fermion(theta))
    I_ = np.zeros(2*N, dtype=complex)
    I_[1::2] = I

    out = np.zeros(N+1, dtype=complex)
    out[:N] = np.fft.fft(I_)[:N] / (2*N) 
    out[-1] = -out[0] - 1.0
    return out
    '''

def t2w_fermion_jump(h, beta):
    N     = len(h)-1
    delta = beta/N
    w  = np.fft.fftfreq(2*N, beta/(np.pi*2*N))[1::2]
    theta = w * delta
    h  = h + 0.5 # account for jump (make continuous)
    h_ = np.concatenate((h, -h[1:]))
    I = 0.5 * delta * W_fermion(theta) * np.fft.ifft(h_[:-1])[1::2] * 2*N + 1.0/(1j*w)
    return I[:N//2]
    
def w2t_fermion_jump(I, beta):
    N     = 2*len(I)
    delta = beta/N
    w  = np.fft.fftfreq(2*N, beta/(np.pi*2*N))[1::2]
    theta = w * delta
    I = np.concatenate((I, np.conj(I)[::-1]))    
    I = 2.0*(I - 1.0/(1j*w))/(delta*W_fermion(theta))
    I_ = np.zeros(2*N, dtype=complex)
    I_[1::2] = I
    out = np.zeros(N+1, dtype=complex)
    out[:N] = np.fft.fft(I_)[:N] / (2*N) - 0.5
    out[-1] = -out[0] - 1.0
    return out

def t2w_boson(h, beta, axis):
    hshape = np.shape(h)
    dim = len(hshape)

    N     = hshape[axis]-1
    delta = beta/N
    w  = np.fft.fftfreq(N, beta/(np.pi*2*N))
    theta = w*delta

    indices = expand(np.arange(N), axis, dim)
    I = delta * expand(W_boson(theta), axis, dim) * np.fft.ifft(np.take_along_axis(h, indices, axis), axis=axis) * N

    indices = expand(np.arange(N//2+1), axis, dim)
    return np.take_along_axis(I, indices, axis)

def w2t_boson(I, beta, axis):
    Ishape = np.shape(I)
    dim = len(Ishape)

    N = 2*(Ishape[axis]-1)
    delta = beta/N
    w  = np.fft.fftfreq(N, beta/(np.pi*2*N))
    theta = w*delta

    indices = expand(np.arange(N//2-1,0,-1), axis, dim)
    
    I = np.concatenate((I, np.take_along_axis(np.conj(I), indices, axis)), axis)

    I = I / (delta * expand(W_boson(theta), axis, dim))
    
    shape = np.array(np.shape(I))
    shape[axis] = N+1
    out = np.zeros(shape, dtype=complex)
    
    indices = expand(np.arange(N), axis, dim)
    np.put_along_axis(out, indices, np.fft.fft(I, axis=axis)/N, axis)
    indices = expand(np.arange(N,N+1), axis, dim)
    np.put_along_axis(out, indices, np.take_along_axis(out, expand(np.arange(1), axis, dim), axis), axis)
    return out

'''
def t2w_boson(h, beta):
    N     = len(h)-1
    delta = beta/N
    w  = np.fft.fftfreq(N, beta/(np.pi*2*N))
    theta = w*delta
    I = delta * W_boson(theta) * np.fft.ifft(h[:-1]) * N
    return I[:N//2+1]

def w2t_boson(I, beta):
    N = 2*(len(I)-1)
    delta = beta/N
    w  = np.fft.fftfreq(N, beta/(np.pi*2*N))
    theta = w*delta
    I = np.concatenate((I, np.conj(I)[-2:0:-1]))
    I = I / (delta * W_boson(theta))
    out = np.zeros(N+1, dtype=complex)
    out[:N] = np.fft.fft(I)/N
    out[-1] = out[0]
    return out
'''

def t2w_copy(h, beta):
    Ntau = len(h)-1
    Nfft = 2*Ntau
    dtau = beta/Ntau

    Cij = -(h[Ntau] + h[0])

    work_tau = np.zeros(2*Ntau, dtype=complex)
    work_tau[Ntau:] =  h[:Ntau]
    work_tau[:Ntau] = -h[:Ntau]

    work_w = np.fft.ifft(work_tau) * 2*Ntau
                        
    X_w = np.zeros(Ntau, dtype=complex)
    w_  = np.fft.fftfreq(2*Ntau, beta/(np.pi*2*Ntau))
    wms = np.zeros(Ntau)
    
    for m in range(Ntau):
        X_w[m] = -0.5 * work_w[2*m+1]
        wm     = w_[2*m+1]

        #if m<Ntau//2:
        #    print(2*np.pi/beta*(m+1-0.5), wm)
        #else:
        #    print(-2*np.pi/beta*(Ntau-m-0.5), wm)
        
        wdt = wm*dtau
        w2dt = wdt*wm
        ez = np.exp(-1j*wm*dtau)
        cs = 4.0*np.sin(0.5*wdt)**2/w2dt
        Cjump = Cij*(1.0/(1j*wm) - (ez-1.0)/w2dt)
        X_w[m] = Cjump + cs*X_w[m]
        wms[m] = wm

    #wms, X_w = np.roll(wms, -Ntau//2), np.roll(X_w, -Ntau//2)
    wms, X_w = wms[:Ntau//2], X_w[:Ntau//2]
    return X_w



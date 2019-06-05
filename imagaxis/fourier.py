import numpy as np
from matplotlib.pyplot import *

def W(t):
    N = len(t)
    #print(t[np.argmin(np.abs(t))])
    if np.abs(t[np.argmin(np.abs(t))])<1e-10:
        out = np.zeros(len(t), dtype=complex)
        out[0] = 1.0
        out[1:] = 2.0*(1.0-np.cos(t[1:]))/t[1:]**2
        return out
    return 2.0*(1.0-np.cos(t))/t**2

def alpha0(t):
    return -(1.0-np.cos(t))/t**2 + 1j*(t-np.sin(t))/t**2

def w2t(x, beta, kind=None):
    if kind=='fermion':
        return w2t_fermion_alpha0(x, beta)
    elif kind=='boson':
        return w2t_boson(x, beta)

def t2w(x, beta, kind=None):
    if kind=='fermion':
        return t2w_fermion_alpha0(x, beta)
    else:
        return t2w_boson(x, beta)

def t2w_fermion_alpha0(h, beta):
    N     = len(h)-1
    delta = beta/N
    w  = np.fft.fftfreq(2*N, beta/(np.pi*2*N))[1::2]
    theta = w*delta
    h_ = np.concatenate((h[:-1], -h[:-1]))
    I = W(theta) * 0.5 * np.fft.ifft(h_)[1::2]*2*N + alpha0(theta)*(h[0]+h[-1])
    I *= delta
    return I[:N//2]

def w2t_fermion_alpha0(I, beta):
    # assume jump is -1
    
    N     = 2*len(I)
    delta = beta/N
    w  = np.fft.fftfreq(2*N, beta/(np.pi*2*N))[1::2]
    theta = w*delta

    # extend to negative frequencies
    I = np.concatenate((I, np.conj(I)[::-1]))
    
    I = 2.0*(I - alpha0(theta)*(-1)*delta)/(delta*W(theta))
    I_ = np.zeros(2*N, dtype=complex)
    I_[1::2] = I

    out = np.zeros(N+1, dtype=complex)
    out[:N] = np.fft.fft(I_)[:N] / (2*N) 
    out[-1] = -out[0] - 1.0
    return out

def t2w_fermion_jump(h, beta):
    N     = len(h)-1
    delta = beta/N
    w  = np.fft.fftfreq(2*N, beta/(np.pi*2*N))[1::2]
    theta = w * delta
    h  = h + 0.5 # account for jump (make continuous)
    h_ = np.concatenate((h, -h[1:]))
    I = 0.5 * delta * W(theta) * np.fft.ifft(h_[:-1])[1::2] * 2*N + 1.0/(1j*w)
    return I[:N//2]
    
def w2t_fermion_jump(I, beta):
    N     = 2*len(I)
    delta = beta/N
    w  = np.fft.fftfreq(2*N, beta/(np.pi*2*N))[1::2]
    theta = w * delta
    I = np.concatenate((I, np.conj(I)[::-1]))    
    I = 2.0*(I - 1.0/(1j*w))/(delta*W(theta))
    I_ = np.zeros(2*N, dtype=complex)
    I_[1::2] = I
    out = np.zeros(N+1, dtype=complex)
    out[:N] = np.fft.fft(I_)[:N] / (2*N) - 0.5
    out[-1] = -out[0] - 1.0
    return out

def t2w_boson(h, beta):
    N     = len(h)-1
    delta = beta/N
    w  = np.fft.fftfreq(N, beta/(np.pi*2*N))
    theta = w*delta
    I = delta * W(theta) * np.fft.ifft(h[:-1]) * N
    return I[:N//2+1]

def w2t_boson(I, beta):
    N = 2*(len(I)-1)
    delta = beta/N
    w  = np.fft.fftfreq(N, beta/(np.pi*2*N))
    theta = w*delta
    I = np.concatenate((I, np.conj(I)[-2:0:-1]))
    I = I / (delta * W(theta))
    out = np.zeros(N+1, dtype=complex)
    out[:N] = np.fft.fft(I)/N
    out[-1] = out[0]
    return out

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



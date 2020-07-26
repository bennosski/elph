
import numpy as np
import matplotlib.pyplot as plt

# 1 dimensional selfenergy and lambm
# for constant density of states (linear band)

g = 1
omega = 1
dw = 0.010
idelta = 0.020j
beta = 4

w = np.arange(-6, 6, dw)
nw = len(w)
nk = 4000
ek = np.linspace(-16, 16, nk)

w_  = w[None,:]
ek_ = ek[:,None]

#---------------------------------
# compute DOS

dos = -1/np.pi * np.mean(1 / (w_ - ek_ + idelta), axis=0).imag

plt.figure()
plt.plot(w, dos)
plt.xlim(-3, 3)

print('dos = ', dos[len(w)//2])

# DOS = 1/8
# so lamb = 2 g^2 DOS / omega = 1/4

lamb = 2 * g**2 * dos / omega

#----------------------------------
# analytic solution at T=0

SRexact = lamb * omega / 2 * np.log(np.abs((omega - w)/(omega + w)))


#----------------------------------
# numerical solution

nB = 1/(np.exp(beta*omega) - 1)
nF_ = 1/(np.exp(beta*ek_) + 1)

SR = g**2/nk * np.sum((nB + nF_)/(w_ - ek_ + omega + idelta) +
                  (nB + 1 - nF_)/(w_ - ek_ - omega + idelta), axis=0)

#plt.figure()

plt.figure()
plt.plot(w, SRexact)
plt.plot(w, SR.real)
plt.xlim(-6, 6)
plt.ylim(-1, 1)

plt.figure()
#plt.plot(w, SRexact)
plt.plot(w, SR.imag)
plt.xlim(-6, 6)
plt.ylim(-1, 1)


h = len(w)//2
wm = np.concatenate((w[:h], w[h+1:]))
SRm = np.concatenate((SR[:h], SR[h+1:]))
SRexactm = np.concatenate((SRexact[:h], SRexact[h+1:]))

plt.figure()
plt.plot(wm, -SRexactm/wm)
plt.plot(wm, -SRm.real/wm)
plt.ylim(0, 1)

print('lamb m from SRexact = ', -(SRexact[h+1]-SRexact[h-1])/(w[h+1]-w[h-1]))
print('lamb from SR = ', -(SR[h+1]-SR[h-1])/(w[h+1]-w[h-1]))




import matplotlib
#matplotlib.use('TkAgg')
from matplotlib.pyplot import *
import numpy as np


#basedir = '/home/groups/simes/bln/data/elph/imagaxis/example/'

#savedir = np.load('savedir.npy')[0]
#print('savedir', savedir)


"""
savedirs = list(np.load('savedirs.npy'))
for i,s in enumerate(savedirs):
    print(i,s)
i = int(input('which dir?'))
savedir = savedirs[i]
"""

savedir = np.load('savedir.npy')[0]
print(savedir)

SR = np.load(savedir+'SR.npy')
PIR = np.load(savedir+'PIR.npy')
DR = np.load(savedir+'DR.npy')
GR = np.load(savedir+'GR.npy')
w  = np.load(savedir+'w.npy')
nk = np.load(savedir+'nk.npy')[0]
dim = np.load(savedir+'dim.npy')[0]
sc  = np.load(savedir+'sc.npy')[0]

if dim==1 and sc:
    print(np.shape(SR))
    print(np.shape(w))

    figure()
    plot(w, SR[nk//4,:,0,0].imag)
    plot(w, SR[nk//4,:,0,1].real)
    title('SR')

    A = -1/np.pi * GR[...,0,0].imag
    figure()
    imshow(A.T, origin='lower', extent=[-np.pi,np.pi,w[0],w[-1]])
    colorbar()
    show()

if dim==1 and not sc:
    A = -1.0/np.pi*GR.imag

    figure()
    plot(w, A[nk//2,:])
    title('A(0)')
    ylim(0, 3.0)
    xlim(-3, -1.5)
    savefig('figs/gkq/A.png')

    figure()
    imshow(A.T, origin='lower', aspect='auto', extent=[-np.pi,np.pi,w[0],w[-1]], vmax=2.5)
    colorbar()
    title('Akw all k')
    savefig('figs/gkq/Ak.png')

    B = -1.0/np.pi*DR.imag

    figure()
    imshow(B.T, origin='lower', aspect='auto', extent=[-np.pi,np.pi,w[0],w[-1]])
    ylim(0, 0.2)
    colorbar()
    title('Bkw all k')
    savefig('figs/gkq/Bk.png')

    figure()
    plot(w, SR[nk//2].imag)
    plot(w, SR[nk//2].real)
    title('SR')
    xlim(-3,-1)
    savefig('figs/gkq/SR.png')

    exit()

    figure()
    plot(w, PIR[nk//2].imag)
    plot(w, PIR[nk//2].real)
    title('PIR')
    show()
 
    exit()


if dim==2 and sc:
    A = -1.0/np.pi*GR[:,:,:,0,0].imag
    figure()
    plot(w, A[nk//4,nk//4])
    title('A')
    ylim(-0.02, 3.0)
    xlim(-3, 3)
    show()

    figure()
    imshow(A[nk//4].T, origin='lower', aspect='auto', extent=[-np.pi,np.pi,w[0],w[-1]])
    colorbar()
    title('GR all k')
    show()

if dim==2 and not sc:
    A = -1.0/np.pi*GR.imag
    figure()
    plot(w, A[nk//4,nk//4])
    title('A')
    ylim(-0.02, 3.0)
    xlim(-3, 3)
    show()

    exit()

    figure()
    plot(w, SR[nk//4,nk//2].imag)
    plot(w, SR[nk//4,nk//2].real)
    title('SR')
    show()

    figure()
    plot(w, PIR[nk//4,nk//2].imag)
    plot(w, PIR[nk//4,nk//2].real)
    title('PIR')
    show()

    figure()
    imshow(-1.0/np.pi*GR[nk//4].imag.T, origin='lower', aspect='auto', extent=[-np.pi,np.pi,w[0],w[-1]])
    colorbar()
    title('GR all k')
    show()

    # interpolate!!!
    # use RectBivariateSpline

    figure()
    imshow(-1.0/np.pi*DR[nk//4].imag.T, origin='lower', aspect='auto', extent=[-np.pi,np.pi,w[0],w[-1]])
    colorbar()
    title('GR all k')
    show()


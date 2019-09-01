import matplotlib
matplotlib.use('TkAgg')
from matplotlib.pyplot import *
import numpy as np
import imagaxis
from functions import band_square_lattice, mylamb2g0

class figures:

    def plot1(self):
        basedir = '/home/groups/simes/bln/data/elph/imagaxis/example/'

        betas = np.linspace(1.0, 20.0, 20) 
        deltas = []
        
        for ibeta in range(len(betas)):
            params = {}
            params['nw']    = 512
            params['nk']    = 20
            params['t']     = 1.0
            params['tp']    = 0.0
            params['omega'] = 1.0
            params['dens']  = 1.0
            params['renormalized'] = True
            params['sc'] = True
            params['band']  = band_square_lattice
            params['beta']  = betas[ibeta]   # 20.0
            params['dim']   = 2
            W    = 8.0
            lamb = 0.2
            params['g0'] = mylamb2g0(lamb, params['omega'], W)

            params['dw']     = 0.001
            params['wmin']   = -4.1
            params['wmax']   = +4.1
            params['idelta'] = 0.010j

            for key in params:
                setattr(self, key, params[key])

            wr = np.arange(self.wmin,self.wmax,self.dw)
            nr = len(wr)

            savedir = basedir+'data/data_{}_nk{}_abstp{:.3f}_dim{}_g0{:.5f}_nw{}_omega{:.3f}_dens{:.3f}_beta{:.4f}_sc{}/'.format('renormalized' if self.renormalized else 'unrenormalized', self.nk, abs(self.tp), self.dim, self.g0, self.nw, self.omega, self.dens, self.beta, 1 if self.sc else 0)

            
            if not os.path.exists(savedir+'GR.npy'): continue

            A = -1/np.pi * np.load(savedir+'GR.npy')[:,:,:,0,0].imag
            
            figure()
            imshow(A[self.nk//4].T, origin='lower', aspect='auto', extent=[-np.pi,np.pi,self.wmin,self.wmax])
            colorbar() 
            title('Akw beta=%1.1f'%self.beta)
            savefig('figs/spec_{}_{:1.8f}.png'.format('R' if self.renormalized else 'U', self.beta))
            
            S = np.load(savedir+'SR.npy')
            deltas.append(np.mean(S[:,:,nr//2,0,1].real))
            print(deltas)
    
            figure()
            plot(wr, S[self.nk//4,self.nk//4,:,0,0].real)
            plot(wr, S[self.nk//4,self.nk//4,:,0,0].imag)
            title('S for beta=%1.1f'%self.beta)
            savefig('figs/S_{}_{:1.8f}.png'.format('R' if self.renormalized else 'U', self.beta)) 
            

        figure()
        plot(1./betas, deltas, '.-')
        ylabel('delta')
        xlabel('T')
        savefig('figs/deltas_{}.png'.format('R' if self.renormalized else 'U'))                



f = figures()
f.plot1()

exit()


#savedir = np.load('savedir.npy')[0]
#print('savedir', savedir)

savedirs = list(np.load('savedirs.npy'))
for i,s in enumerate(savedirs):
    print(i,s)
i = int(input('which dir?'))
savedir = savedirs[i]

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
    plot(w, SR[nk//2].imag)
    plot(w, SR[nk//2].real)
    title('SR')
    show()

    figure()
    plot(w, PIR[nk//2].imag)
    plot(w, PIR[nk//2].real)
    title('PIR')
    show()

    figure()
    imshow(A.T, origin='lower', aspect='auto', extent=[-np.pi,np.pi,w[0],w[-1]])
    colorbar()
    title('GR all k')
    show()

    figure()
    imshow(-1.0/np.pi*DR.imag.T, origin='lower', aspect='auto', extent=[-np.pi,np.pi,w[0],w[-1]])
    colorbar()
    title('GR all k')
    show()

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


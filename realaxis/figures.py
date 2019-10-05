import matplotlib
#matplotlib.use('TkAgg')
from matplotlib.pyplot import *
import numpy as np
import imagaxis
from functions import *
import os

# goals:
# forward scattering plots : data is there I think
#     - shows difference in forward scattering
# 2D SC plots : need to double check, but probably close
#     - shows enhancement of SC in renormalized case (why?)
#
# CDW phase stuff?
#    - todo add dispersion to phonon
#    - ek+q array
#    - right taus...


class figures:

    def forward_scattering(self):

        params = {}
        params['nw']    = 512
        params['nk']    = 400
        params['t']     = 1.0
        params['tp']    = 0.0
        params['omega'] = 0.14
        params['dens']  = 1.0
        params['sc']    = 0
        params['band']  = band_1d_lattice
        params['beta']  = 80.0
        params['dim']   = 1

        params['dw']     = 0.001
        params['wmin']   = -4.1
        params['wmax']   = +4.1
        params['idelta'] = 0.005j

        basedir = '/home/groups/simes/bln/data/elph/imagaxis/forward/'
        if not os.path.exists(basedir): os.makedirs(basedir)

        lamb = 1.0
        W    = 8.0
        params['g0'] = mylamb2g0(lamb, params['omega'], W)
        params['q0'] = 0.05
        params['gq2'] = gexp_1d(params['nk'], params['q0'])**2
        print('g0 is ', params['g0'])

        if not os.path.exists('figs/forward'):
            os.mkdir('figs/forward')

        savedirs = []
        for renormalized in (False, True):
            params['renormalized'] = renormalized

            for key in params:
                setattr(self, key, params[key])

            wr = np.arange(self.wmin,self.wmax,self.dw)
            nr = len(wr)

            savedir = basedir+'data/data_{}_nk{}_abstp{:.3f}_dim{}_g0{:.5f}_nw{}_omega{:.3f}_dens{:.3f}_beta{:.4f}_sc{}/'.format('renormalized' if self.renormalized else 'unrenormalized', self.nk, abs(self.tp), self.dim, self.g0, self.nw, self.omega, self.dens, self.beta, 1 if self.sc else 0)
            savedirs.append(savedir)

        f = figure()
        f.set_size_inches(15, 5)

        ax = f.add_axes([0.12, 0.72, 0.8, 0.26])
        savedir = savedirs[0]
        GR = np.load(savedir+'GR.npy')
        A  = -1.0/np.pi * GR.imag
        ax.plot(wr, A[self.nk//2])
        ax.set_xlim(-2.6, -1.5)
        ax.set_ylim(0, 4)
        ax.get_xaxis().set_visible(False)
         
        ax = f.add_axes([0.12, 0.44, 0.8, 0.26])
        savedir = savedirs[1]
        GR = np.load(savedir+'GR.npy')
        A  = -1.0/np.pi * GR.imag
        ax.plot(wr, A[self.nk//2])
        ax.set_xlim(-2.6, -1.5)
        ax.set_ylim(0, 4)

        ax = f.add_axes([0.12, 0.1, 0.8, 0.26])
        savedir = savedirs[1]
        DR = np.load(savedir+'DR.npy')
        B  = -1.0/np.pi * DR.imag
        imshow(B.T, origin='lower', aspect='auto', extent=[-np.pi,np.pi,self.wmin,self.wmax], vmin=0, vmax=25)
        colorbar() 
        ylim(0, 0.25)
        xlim(-1, 1)
  
        savefig('figs/forward/Aw_{}_g0{:.8f}_q0{:.8f}.png'.format('R' if self.renormalized else 'U', self.g0, self.q0))

        for i,savedir in enumerate(savedirs):
            GR = np.load(savedir+'GR.npy')
            A  = -1.0/np.pi * GR.imag

            plot(wr, A[self.nk//2])
            xlim(-2.6, -1.5)
            ylim(0, 4)
            title('Aw')
            savefig('figs/forward/Aw_{}_g0{:.8f}_q0{:.8f}.png'.format('R' if self.renormalized else 'U', self.g0, self.q0))

            figure()
            imshow(A.T, origin='lower', aspect='auto', extent=[-np.pi,np.pi,self.wmin,self.wmax])
            colorbar() 
            title('Akw beta=%1.1f'%self.beta)
            savefig('figs/forward/spec_{}_g0{:.8f}_q0{:.8f}.png'.format('R' if self.renormalized else 'U', self.g0, self.q0))

            DR = np.load(savedir+'DR.npy')
            B  = -1.0/np.pi * DR.imag

            figure()
            imshow(B.T, origin='lower', aspect='auto', extent=[-np.pi,np.pi,self.wmin,self.wmax], vmin=0, vmax=25)
            colorbar() 
            ylim(0, 0.25)
            xlim(-1, 1)
            title('Bkw')
            savefig('figs/forward/B_{}_g0{:.8f}_q0{:.8f}.png'.format('R' if self.renormalized else 'U', self.g0, self.q0))

    #------------------------------------------------------------
    def sc2d(self, renorm, lamb, dens, tp):
        # 2D SC plots vs temperature

        basedir = '/home/groups/simes/bln/data/elph/imagaxis/example/'

        betas = np.linspace(1.0, 40.0, 40) 
        deltas = []
        Ts = []
        
        '''
        renorm = False
        lamb = 0.3
        dens = 1.0
        tp = 0.0
        '''

        for ibeta in range(len(betas)):
            params = {}
            params['nw']    = 512
            params['nk']    = 20
            params['t']     = 1.0
            params['tp']    = tp
            params['omega'] = 1.0
            params['dens']  = dens
            params['renormalized'] = renorm
            params['sc'] = True
            params['band']  = band_square_lattice
            params['beta']  = betas[ibeta]   # 20.0
            params['dim']   = 2
            W    = 8.0
            params['g0'] = mylamb2g0(lamb, params['omega'], W)
            params['dw']     = 0.001
            params['wmin']   = -4.1
            params['wmax']   = +4.1
            params['idelta'] = 0.010j

            for key in params:
                setattr(self, key, params[key])

            wr = np.arange(self.wmin,self.wmax,self.dw)
            nr = len(wr)

            svdir = 'figs/sc2d_{}_g0{:.5f}_dens{:.2f}/'.format('R' if self.renormalized else 'U', self.g0, self.dens)
            if not os.path.exists(svdir): os.mkdir(svdir)

            savedir = basedir+'data/data_{}_nk{}_abstp{:.3f}_dim{}_g0{:.5f}_nw{}_omega{:.3f}_dens{:.3f}_beta{:.4f}_sc{}/'.format('renormalized' if self.renormalized else 'unrenormalized', self.nk, abs(self.tp), self.dim, self.g0, self.nw, self.omega, self.dens, self.beta, 1 if self.sc else 0)
            
            print('ibeta', ibeta)

            if not os.path.exists(savedir+'GR.npy'):
                print('no GR :(')
                continue

            print('found data')

            A = -1/np.pi * np.load(savedir+'GR.npy')[:,:,:,0,0].imag
            
            figure()
            imshow(A[self.nk//4].T, origin='lower', aspect='auto', extent=[-np.pi,np.pi,self.wmin,self.wmax])
            colorbar() 
            title('Akw beta=%1.1f'%self.beta)
            savefig(svdir+'spec_{}_{:1.8f}.png'.format('R' if self.renormalized else 'U', self.beta))
            close()

            S = np.load(savedir+'SR.npy')
            Ts.append(1/betas[ibeta])
            deltas.append(np.mean(S[:,:,nr//2,0,1].real))
            print(deltas)
    
            figure()
            plot(wr, S[self.nk//4,self.nk//4,:,0,0].real)
            plot(wr, S[self.nk//4,self.nk//4,:,0,0].imag)
            title('S for beta=%1.1f'%self.beta)
            savefig(svdir+'S_{}_{:1.8f}.png'.format('R' if self.renormalized else 'U', self.beta)) 
            close()

            figure()
            plot(wr, S[self.nk//4,self.nk//4,:,0,1].real)
            plot(wr, S[self.nk//4,self.nk//4,:,0,1].imag)
            title('W for beta=%1.1f'%self.beta)
            savefig(svdir+'W_{}_{:1.8f}.png'.format('R' if self.renormalized else 'U', self.beta)) 
            close()

        figure()
        plot(Ts, np.abs(np.array(deltas)), '.-')
        ylabel('delta')
        xlabel('T')
        savefig(svdir+'deltas_{}.png'.format('R' if self.renormalized else 'U'))                
        close()

f = figures()

for renorm in (True, False):
    for lamb in (0.2, 0.3):
        for dens,tp in zip([0.8, 1.0], [-0.3, 0.0]):
            f.sc2d(renorm, lamb, dens, tp)

#f.forward_scattering()


#-----------------------------------------------------------
#----------------------------------------------------------
# Old code below

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


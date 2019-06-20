import os
from numpy import *
import matplotlib
matplotlib.use('agg')
from matplotlib.pyplot import *
import fourier

def load_all(folder):
    g0 = load(folder+'/g0.npy')[0]
    omega = load(folder+'/omega.npy')[0]
    beta  = load(folder+'/beta.npy')[0]
    dens = load(folder+'/dens.npy')[0]
    nk = load(folder+'/nk.npy')[0]
    nw = load(folder+'/nw.npy')[0]
    S = load(folder+'/S.npy')
    PI = load(folder+'/PI.npy')
    G = load(folder+'/G.npy')
    D = load(folder+'/D.npy')

    S = apply_along_axis(fourier.t2w, 2, S, beta, 'fermion')
    PI = apply_along_axis(fourier.t2w, 2, PI, beta, 'boson')
    G = apply_along_axis(fourier.t2w, 2, G, beta, 'fermion')
    D = apply_along_axis(fourier.t2w, 2, D, beta, 'fermion')

    return g0, omega, beta, dens, nk, nw, S, PI, G, D

def analyze_G():
    folder = os.listdir('data/')
    
    g0, omega, beta, dens, nk, nw, S, PI, G, D = load_all('data/'+folder[0])
 
    print('shape S', shape(S))
  
    figure()
    G0 = G.copy()
    Gtau = fourier.w2t_fermion_alpha0(G0[nk//2,nk//2,:], beta)
    taus = linspace(0, beta, len(Gtau))
    plot(taus, Gtau.real)

    G0 = G.copy()
    Gtau = fourier.w2t_fermion_alpha0(G0[nk//2,0,:], beta)
    plot(taus, Gtau.real)

    G0 = G.copy()
    Gtau = fourier.w2t_fermion_alpha0(G0[0,0,:], beta)
    plot(taus, Gtau.real)

    legend(['G', 'X', 'M'])

    savefig('Gtau')
    show()

#analyze_G()


def analyze_single_particle(base_dir):
    #base_dir = 'data_single_particle/'
    folders = os.listdir(base_dir)

    lims = [0.06, 0.25, 3]

    for i,folder in enumerate(folders):
        print(folder)

        g0, omega, beta, dens, nk, nw, S, PI, G, D = load_all(base_dir + folder)
        print('g0 = ', g0)
        lamb = 2.4*2.0*g0**2/(omega * 8.0)
        print('lamb ilya = ', lamb)

        print('shape S', shape(S))

        wn = (2*arange(nw)+1)*pi/beta

        figure()
        plot(wn, -S[nk//2+3, nk//2+2, :].imag, '.-', color='orange')
        plot(wn, -S[0, nk//2+1, :].imag, '.-', color='green')
        xlim(0, 2)
        ylim(0, lims[i])
        legend(['34deg', '9deg'])
        title(f'lamb = {lamb:1.1f}')
        savefig(f'imag S {lamb:1.1f}.png')    
    
    figure()
    ls = []
    for folder in folders:
        print(folder)
        g0, omega, beta, dens, nk, nw, S, PI, G, D = load_all(base_dir + folder)
        print('g0 = ', g0)
        lamb = 2.4*2.0*g0**2/(omega * 8.0)

        print('lamb ilya = ', lamb)

        print('shape PI', shape(PI))

        wn = (2*arange(nw)+1)*pi/beta

        y = sqrt(omega**2 + 2*omega*PI[:,:,0]) / omega
        diag = [y[i,i] for i in range(1, nk//2+1)]
        yall = concatenate((y[nk//2::-1,nk//2], y[0,nk//2-1::-1], diag))
        yall = yall[1:-1]
        plot(yall, '.-')
        ylim(0, 1)
        ls.append('lamb=%.1f'%lamb)
        
    legend(ls)
    savefig('omega')

analyze_single_particle('data/')



def analyze_x_vs_T():
    folders = os.listdir('data_x_vs_T/')

    lambs = []
    xs = []
    xcdws = []
    for folder in folders:
        print(folder)
        xs.append(load('data/'+folder+'/Xsc.npy')[0])
        lambs.append(load('data/'+folder+'/lamb.npy')[0])
        xcdws.append(np.argmax(load('data/'+folder+'/Xcdw.npy')))

    lambs, xs, xcdws = zip(*sorted(zip(lambs, xs, xcdws)))

    print(xs)
    print('')
    print(2.4*array(lambs))

    figure()
    plot(2.4*array(lambs), xs)
    plot(2.4*array(lambs), array(xcdws)/300.0)
    ylim(0, 1.0)
    savefig('xsc')

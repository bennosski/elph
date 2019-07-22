import os
from numpy import *
import matplotlib
matplotlib.use('agg')
from matplotlib.pyplot import *
import fourier
from params import g02lamb
from collections import defaultdict
from scipy.stats import linregress
import shutil

def load_all(folder):
    res = defaultdict(lambda : None)

    quantities = ['g0', 'omega', 'beta', 'dens', 'nk', 'nw', 'tp', 'G', 'D', 'S', 'PI', 'Xsc', 'Xcdw']

    for x in quantities:
        try:
            y = load(folder+'/%s.npy'%x)
            if len(y)==1:
                res[x] = y[0]
            else:
                res[x] = y
        except:
            print('missing %s'%x)
    return res

def analyze_G():
    folder = os.listdir('data/')
    
    res = load_all('data/'+folder[0])
    g0, omega, beta, dens, nk, nw, tp, S, PI, G, D = res['g0'], res['omega'], res['beta'], res['dens'], res['nk'], res['nw'], res['tp'], res['S'], res['PI'], res['G'], res['D']
 
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


def analyze_single_particle(basedir):
    #base_dir = 'data_single_particle/'

    datadir = basedir + 'data/'

    folders = sorted(os.listdir(datadir))

    lims = [0.06, 0.25, 3]

    for i,folder in enumerate(folders):
        print(folder)

        res = load_all(datadir + folder)
        g0, omega, beta, dens, nk, nw, tp, S, PI, G, D = res['g0'], res['omega'], res['beta'], res['dens'], res['nk'], res['nw'], res['tp'], res['S'], res['PI'], res['G'], res['D']

        print('g0 = ', g0)
        lamb = 2.4*2.0*g0**2/(omega * 8.0)
        print('lamb ilya = ', lamb)

        print('shape S', shape(S))

        wn = (2*arange(nw)+1)*pi/beta

        S = fourier.t2w_fermion_alpha0(S, beta, 2)[0]

        figure()
        plot(wn, -S[nk//2+3, nk//2+2, :].imag, '.-', color='orange')
        plot(wn, -S[0, nk//2+1, :].imag, '.-', color='green')
        xlim(0, 2)
        ylim(0, lims[i])
        legend(['34deg', '9deg'])
        title(f'lamb = {lamb:1.1f}')
        savefig(f'{basedir}imag S {lamb:1.1f}.png')    
    
    figure()
    ls = []
    for folder in folders:
        print(folder)
        res = load_all(datadir + folder)
        g0, omega, beta, dens, nk, nw, tp, S, PI, G, D = res['g0'], res['omega'], res['beta'], res['dens'], res['nk'], res['nw'], res['tp'], res['S'], res['PI'], res['G'], res['D']

        print('g0 = ', g0)
        lamb = 2.4*2.0*g0**2/(omega * 8.0)

        print('lamb ilya = ', lamb)

        print('shape PI', shape(PI))

        wn = (2*arange(nw)+1)*pi/beta

        PI = fourier.t2w_boson(PI, beta, 2)

        y = sqrt(omega**2 + 2*omega*PI[:,:,0]) / omega
        diag = [y[i,i] for i in range(1, nk//2+1)]
        yall = concatenate((y[nk//2::-1,nk//2], y[0,nk//2-1::-1], diag))
        yall = yall[1:-1]
        plot(yall, '.-')
        ylim(0, 1)
        ls.append('lamb=%.1f'%lamb)
        
    legend(ls)
    savefig(f'{basedir}omega')

#analyze_single_particle('test/data/')


def analyze_x_vs_lamb(basedir):
    datadir = basedir + 'data/'

    folders = os.listdir(datadir)

    xs = []
    lambs = []
    xcdws = []

    for folder in folders:
        res = load_all(datadir + folder)
        g0, omega, beta, dens, nk, nw, tp, S, PI, G, D, Xsc, Xcdw = res['g0'], res['omega'], res['beta'], res['dens'], res['nk'], res['nw'], res['tp'], res['S'], res['PI'], res['G'], res['D'], res['Xsc'], res['Xcdw']

        print('g0',g0)
        print('omega',omega)

        print(folder)
        xs.append(Xsc)
        W = 8.0
        lambs.append(g02lamb(g0, omega, W))
        xcdws.append(amax(Xcdw))

    lambs, xs, xcdws = zip(*sorted(zip(lambs, xs, xcdws)))

    print(xcdws)
    print('')
    print(array(lambs))

    f = figure()
    f.set_size_inches(8, 5)
    plot(array(lambs), xs, 'g')
    plot(array(lambs), array(xcdws)/300.0, 'orange')
    ylim(0, 1.0)
    xlim(0, 0.65)
    savefig(basedir + 'xsc')

#analyze_x_vs_lamb('test/data_ilya_susceptibilities/')

def get_Tc(basedir):
    
    folders = os.listdir(basedir)

    data = {}

    for folder in folders:

        res = load_all(basedir+folder)
        g0, omega, beta, dens, nk, nw, tp, S, PI, G, D = res['g0'], res['omega'], res['beta'], res['dens'], res['nk'], res['nw'], res['tp'], res['S'], res['PI'], res['G'], res['D']

        #Xs = load(basedir+folder+'/Xs.npy', allow_pickle=True)
        #print(Xs)
       
        Xs = load(basedir+folder+'/Xs.npy', allow_pickle=True).item()
        Xscs = []
        for beta in Xs:
            Xscs.append([beta, Xs[beta]['Xsc']])
        Xscs = array(sorted(Xscs))
        
        
        print('nones')
        notnones = [False if x is None else True for x in Xscs[:,1]]

        Xscs = Xscs[notnones]

        data['%1.2f'%omega] = {}
        data['%1.2f'%omega]['%1.10f'%g0] = Xscs
        
        print('')
        print(omega, g0)
        print(Xscs)

        W = 8
        lamb = g02lamb(g0, omega, W)

        figure()
        plot(1./Xscs[:,0], 1./Xscs[:,1], '.-')
        xlim(0, gca().get_xlim()[1])
        name = '$\Omega$='+'%1.1f'%omega+' lamb=%1.2f'%lamb+' nk=%d'%nk+' tp=%1.1f'%tp
        title(name)

        print(name)
        slope, intercept = linregress(1./Xscs[-2:,0], 1./Xscs[-2:,1])[:2]
        print(slope, intercept)
        Tc = -intercept / slope
        print('Tc', Tc)
        print('Tc/omega', Tc/omega)
        x = linspace(0, amax(1./Xscs[-1,0]), 10)
        plot(x, slope*x + intercept)

        name = name.replace('.', 'p')
        savefig('figs/inv xsc %s'%name)

#get_Tc('data/')


def analyze_Tc(basedir_nambu, basedir_normal):
    
    basedir = basedir_nambu
    folders = os.listdir(basedir+'data/')
    omega = load(basedir+'data/'+folders[0]+'/omega.npy')[0]

    Xs = load(basedir+'Xs.npy', allow_pickle=True).item()
    print(Xs.keys())
    betas = array(sorted(Xs.keys()))
    orders = []
    for beta in betas:
        orders.append(Xs[beta]['Gloc'][0,1])
    orders = array(orders)

    figure()
    plot(1.0/betas, abs(orders*10), '.-')

    #basedir = '/scratch/users/bln/elph/imagaxis/match_Tc/'
    basedir = basedir_normal
    Xs = load(basedir+'Xs.npy', allow_pickle=True).item()
    print(Xs.keys())
    betas = array(sorted(Xs.keys()))
    xscs = []
    for beta in betas:
        xscs.append(Xs[beta]['Xsc'])
    xscs = array(xscs)
    plot(1.0/betas, 1.0/xscs, '.-')

    #ylim(-0.02, gca().get_ylim()[1])
    #xlim(0, 0.3)
    xlim(0, 0.05)
    ylim(-1.0, 1.0)
    
    legend(['10$\Delta$', '1/$X_{sc}$'])
    title('omega = %1.1f'%omega, fontsize=18)
    xlabel('T', fontsize=18)

    savefig(basedir_nambu + 'Tc')
    savefig(basedir_normal + 'Tc')

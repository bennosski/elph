import os
from numpy import *
import matplotlib
matplotlib.use('agg')
from matplotlib.pyplot import *
import fourier
from params import g02lamb
from collections import defaultdict
from scipy.stats import linregress

def load_all(folder):
    res = defaultdict(lambda : None)

    quantities = ['g0', 'omega', 'beta', 'dens', 'nk', 'nw', 'tp', 'G', 'D', 'S', 'PI']

    for x in quantities:
        try:
            y = load(folder+'/%s.npy'%x)
            if len(y)==1:
                res[x] = y[0]
            else:
                res[x] = y
        except:
            print('missing %s'%x)

    '''
    res['g0'] = load(folder+'/g0.npy')[0]
    res['omega'] = load(folder+'/omega.npy')[0]
    res['beta']  = load(folder+'/beta.npy')[0]
    res['dens'] = load(folder+'/dens.npy')[0]
    res['nk'] = load(folder+'/nk.npy')[0]
    res['nw'] = load(folder+'/nw.npy')[0]

        G = load(folder+'/G.npy')
        D = load(folder+'/D.npy')
        res['G'] = apply_along_axis(fourier.t2w, 2, G, beta, 'fermion')
        res['D'] = apply_along_axis(fourier.t2w, 2, D, beta, 'fermion')
        res['tp'] = load(folder+'/tp.npy')[0]

        res['S'] = load(folder+'/S.npy')
        res['PI'] = load(folder+'/PI.npy')
        res['S'] = apply_along_axis(fourier.t2w, 2, S, beta, 'fermion')
        res['PI'] = apply_along_axis(fourier.t2w, 2, PI, beta, 'boson')

    except:
        res['G'] = None
        res['D'] = None
        res['tp'] = None
    '''

    return res
    #return g0, omega, beta, dens, nk, nw, S, PI, G, D

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


def analyze_single_particle(base_dir):
    #base_dir = 'data_single_particle/'
    folders = os.listdir(base_dir)

    lims = [0.06, 0.25, 3]

    for i,folder in enumerate(folders):
        print(folder)

        res = load_all(base_dir + folder)
        g0, omega, beta, dens, nk, nw, tp, S, PI, G, D = res['g0'], res['omega'], res['beta'], res['dens'], res['nk'], res['nw'], res['tp'], res['S'], res['PI'], res['G'], res['D']

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
        res = load_all(base_dir + folder)
        g0, omega, beta, dens, nk, nw, tp, S, PI, G, D = res['g0'], res['omega'], res['beta'], res['dens'], res['nk'], res['nw'], res['tp'], res['S'], res['PI'], res['G'], res['D']

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

#analyze_single_particle('test/data/')


def analyze_x_vs_T(basedir):
    folders = os.listdir(basedir)

    lambs = []
    xs = []
    xcdws = []
    for folder in folders:
        print(folder)
        xs.append(load(basedir+folder+'/Xsc.npy')[0])
        
        omega = load(basedir+folder+'/omega.npy')[0]
        W = 8.0

        g0 =load(basedir+folder+'/g0.npy')[0]
        lambs.append(g0**2 * 2.4 / (0.5 * omega * W))

        xcdws.append(np.amax(load(basedir+folder+'/Xcdw.npy')))

    lambs, xs, xcdws = zip(*sorted(zip(lambs, xs, xcdws)))

    print(xcdws)
    print('')
    print(array(lambs))

    figure()
    plot(array(lambs), xs)
    plot(array(lambs), array(xcdws)/300.0)
    ylim(0, 1.0)
    xlim(0, 0.65)
    savefig('xsc')

#analyze_x_vs_T('test/data_ilya_susceptibilities/')

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

get_Tc('data/')





import src
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
from functions import read_params, g02lamb_ilya
import fourier
from migdal_2d import Migdal
import matplotlib.pyplot as plt 

#basedir = '/home/groups/tpd/bln/migdal_check_vs_ilya/'
basedir = '/scratch/users/bln/migdal_check_vs_ilya/single_particle/'

def load_all(folder):
    res = defaultdict(lambda : None)

    quantities = ['g0', 'omega', 'beta', 'dens', 'nk', 'nw', 'tp', 'G', 'D', 'S', 'PI', 'Xsc', 'Xcdw']

    for x in quantities:
        try:
            #print(os.path.join(folder, '{}.npy'.format(x)))
            y = np.load(os.path.join(folder, '{}.npy'.format(x)))
            if len(y)==1:
                res[x] = y[0]
            else:
                res[x] = y
        except:
            print('missing %s'%x)
    return res


def analyze_single_particle(basedir):
    #base_dir = 'data_single_particle/'

    datadir = basedir + 'data/'

    folders = sorted(os.listdir(datadir))

    lims = [0.06, 0.3, 1]

    print('folders', folders)

    for i,folder in enumerate(folders):
        print(folder)

        print(datadir + folder)

        res = load_all(datadir + folder)
        g0, omega, beta, dens, nk, nw, tp, S, PI, G, D = res['g0'], res['omega'], res['beta'], res['dens'], res['nk'], res['nw'], res['tp'], res['S'], res['PI'], res['G'], res['D']

        print('g0 = ', g0)
        lamb = 2.4*2.0*g0**2/(omega * 8.0)
        print('lamb ilya = ', lamb)

        print('shape S', np.shape(S))

        wn = (2*np.arange(nw)+1)*np.pi/beta

        S = fourier.t2w_fermion_alpha0(S, beta, 2)[0]

        plt.figure()
        plt.plot(wn, -S[nk//2+3, nk//2+2, :].imag, '.--', color='orange')
        plt.plot(wn, -S[0, nk//2+1, :].imag, '.--', color='green')
        plt.xlim(0, 2)
        plt.ylim(0, lims[i])
        plt.legend(['34deg', '9deg'])
        plt.title(f'lamb = {lamb:1.1f}')
        plt.savefig(f'{basedir}imag S {lamb:1.1f}.png')    

    
    plt.figure()
    ls = []
    for folder in folders:
        print(folder)
        res = load_all(datadir + folder)
        g0, omega, beta, dens, nk, nw, tp, S, PI, G, D = res['g0'], res['omega'], res['beta'], res['dens'], res['nk'], res['nw'], res['tp'], res['S'], res['PI'], res['G'], res['D']

        print('g0 = ', g0)
        lamb = 2.4*2.0*g0**2/(omega * 8.0)

        print('lamb ilya = ', lamb)

        print('shape PI', np.shape(PI))

        wn = (2*np.arange(nw)+1)*np.pi/beta

        PI = fourier.t2w_boson(PI, beta, 2)

        y = np.sqrt(omega**2 + 2*omega*PI[:,:,0]) / omega
        diag = [y[i,i] for i in range(1, nk//2+1)]
        yall = np.concatenate((y[nk//2::-1,nk//2], y[0,nk//2-1::-1], diag))
        yall = yall[1:-1]
        plt.plot(yall, '.-')
        plt.ylim(0, 1)
        ls.append('lamb=%.1f'%lamb)
        
    plt.legend(ls)
    plt.savefig(f'{basedir}omega')

analyze_single_particle(basedir)


#-------------------------------------------------------------
# compare renormalized and unrenormalized
print('comparing renormalized and unrenormalized')

basedir = '/scratch/users/bln/migdal_check_vs_ilya/single_particle/'
folder = 'data_renormalized_nk12_abstp0.300_dim2_g00.33665_nw512_omega0.170_dens0.800_beta16.0000_QNone/'

params = read_params(basedir, folder)
lamb = g02lamb_ilya(params['g0'], params['omega'], 8)
print('lambda = ', lamb)

S = np.load(basedir + 'data/' + folder + 'S.npy')
wn = (2*np.arange(params['nw'])+1)*np.pi/params['beta']
S = fourier.t2w(S, params['beta'], 2, 'fermion')[0]
nk = params['nk']


basedir = '/scratch/users/bln/migdal_check_vs_ilya/single_particle_unrenorm/'
folder = 'data_unrenormalized_nk12_abstp0.300_dim2_g00.33665_nw512_omega0.170_dens0.800_beta16.0000_QNone/'

uS = np.load(basedir + 'data/' + folder + 'S.npy')
uS = fourier.t2w(uS, params['beta'], 2, 'fermion')[0]

plt.figure()
plt.plot(wn, -S[nk//2+3, nk//2+2, :].imag, '.--', color='orange')
plt.plot(wn, -S[0, nk//2+1, :].imag, '.--', color='green')
plt.plot(wn, -uS[0, 0, :].imag, '.--', color='red')
plt.xlim(0, 2)
plt.ylim(0, 0.3)
plt.ylabel('Im S(iwn)')
plt.xlabel('wn')
plt.legend(['RME 34deg', 'RME 9deg', 'UME'])
plt.title(f'lamb = {lamb:1.1f}')
plt.savefig(f'{basedir}imag S compare.png')    


# -------------------------------------------------------------
# plot x vs lamb

basedir = '/scratch/users/bln/migdal_check_vs_ilya/susceptibilities/'
lambs = np.load(basedir + 'lambs_0.8_omega0.170000.npy')
xscs  = np.load(basedir + 'xscs_0.8_omega0.170000.npy')
xcdws = np.load(basedir + 'xcdws_0.8_omega0.170000.npy')


plt.figure()
plt.plot(lambs, xscs, '.-')
plt.plot(lambs, xcdws/80, '.-')
plt.ylim(0, 1)
plt.xlim(0, 0.55)
plt.savefig(basedir + 'migdal_check')

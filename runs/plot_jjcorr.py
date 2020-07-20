import src
from migdal_2d import Migdal
from real_2d import RealAxisMigdal
from functions import read_params, band_square_lattice
import numpy as np
import matplotlib.pyplot as plt

basedir = '/scratch/users/bln/elph/data/2d/'
rfolder = 'data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone/'
ufolder = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone/'

def run_real_axis():
    params = read_params(basedir, rfolder)
    params['band'] = band_square_lattice
    migdal = RealAxisMigdal(params, basedir)
    G  = np.load(basedir + 'data/' + rfolder + 'G.npy')
    GR = np.load(basedir + 'data/' + rfolder + 'GR.npy')
    migdal.compute_jjcorr(G, GR)


    params = read_params(basedir, ufolder)
    params['band'] = band_square_lattice
    migdal = RealAxisMigdal(params, basedir)
    G  = np.load(basedir + 'data/' + ufolder + 'G.npy')
    GR = np.load(basedir + 'data/' + ufolder + 'GR.npy')
    migdal.compute_jjcorr(G, GR)



def run_imag_axis():
    params = read_params(basedir, rfolder)
    migdal = Migdal(params, basedir)
    G = np.load(basedir + 'data/' + rfolder + 'G.npy')
    D = np.load(basedir + 'data/' + rfolder + 'D.npy')
    print(G.shape)
    migdal.compute_jjcorr(G, D)

    params = read_params(basedir, ufolder)
    migdal = Migdal(params, basedir)
    G = np.load(basedir + 'data/' + ufolder + 'G.npy')
    D = np.load(basedir + 'data/' + ufolder + 'D.npy')
    print(G.shape)
    migdal.compute_jjcorr(G, D)



def plot_real_axis():
    jjwu = np.load(basedir + 'data/' + ufolder + 'jjw.npy')
    jjwr = np.load(basedir + 'data/' + rfolder + 'jjw.npy')

    params = read_params(basedir, rfolder)
    params['band'] = band_square_lattice
    migdal = RealAxisMigdal(params, basedir)
    wn, vn, ek, w, nB, nF, DRbareinv = migdal.setup_realaxis()

    nr = len(w)
    print('nr', nr)
    print('len(w)', len(w))

    '''
    plt.figure()
    plt.plot(w[nr//2+1:], jjw[nr//2+1:].real)
    plt.savefig('jjw_re')

    plt.figure()
    plt.plot(w[nr//2+1:], jjw[nr//2+1:].imag)
    plt.savefig('jjw_im')

    plt.figure()
    plt.plot(w[nr//2+1:], 1/w[nr//2+1:]*jjw[nr//2+1:].real)
    plt.savefig('cond_re')
    '''

    plt.figure()
    plt.plot(w[nr//2+1:], -1/w[nr//2+1:]*jjwu[nr//2+1:].imag)
    plt.plot(w[nr//2+1:], -1/w[nr//2+1:]*jjwr[nr//2+1:].imag)
    plt.legend(['unrenormalized ME', 'renormalized ME'])
    plt.ylabel('$\sigma(\omega)$', fontsize=13)
    plt.xlabel('$\omega$', fontsize=13)
    plt.savefig('cond')


def plot_imag_axis():

    '''
    plt.figure()
    plt.plot(jj0t.real)
    plt.plot(jj0t.imag)
    plt.ylabel('Re $\Lambda^0(\tau)$')
    #plt.title('jj0tau')
    plt.savefig('jj0tau{}'.format('r' if self.renormalized else 'u'))
    '''

    jjvu = np.load(basedir + 'data/' + ufolder + 'jjv.npy')
    jjvr = np.load(basedir + 'data/' + rfolder + 'jjv.npy')
    params = read_params(basedir, rfolder)
    ivn = 2*np.arange(len(jjvr))*np.pi/params['beta']

    plt.figure()
    plt.plot(ivn, jjvu.real, '.-')
    plt.plot(ivn, jjvr.real, '.-')
    plt.legend(['unrenormalized ME', 'renormalized ME'])
    plt.ylabel('Re ' + r'$\Lambda (i \nu_n)$', fontsize=13)
    plt.xlabel(r'$\nu_n$', fontsize=13)
    plt.xlim(0, 2.5)
    plt.savefig('jjv')


run_imag_axis()
run_real_axis()
plot_imag_axis()
plot_real_axis()


'''
uDR = np.load(basedir + 'data/' + ufolder + 'DR.npy')
DR = np.load(basedir + 'data/' + folder + 'DR.npy')
print(DR.shape)
nw = DR.shape[2]
nk = DR.shape[0]

PIR = np.load(basedir + 'data/' + folder + 'PIR.npy')
icdw = np.argmax(np.abs(PIR[:,:,nw//2]).real.flatten())
ikx, iky = np.unravel_index(icdw, (nk, nk))

w = np.load(basedir + 'data/' + folder + 'w.npy')

plt.figure()
plt.plot(w, -1/np.pi*DR[nk//2,nk//2,:].imag)
plt.plot(w, -1/np.pi*DR[0,0,:].imag)
plt.plot(w, -1/np.pi*DR[ikx,iky,:].imag)
plt.plot(w, -1/np.pi*uDR[0,0,:].imag)
plt.legend(['q=0', 'q=(pi,pi)', 'q for max PI(q)', 'unrenormalized case for reference'])
plt.xlim(0, 0.2)
y1,y2 = plt.gca().get_ylim()
plt.ylim(0, y2)
plt.xlabel('$\omega$', fontsize=14)
plt.ylabel('B(q, $\omega$)', fontsize=14)
plt.savefig(basedir + 'phonon_width')
plt.close()
'''

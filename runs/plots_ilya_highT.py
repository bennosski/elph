
import os
import src
from functions import mylamb2g0, band_square_lattice
from real_2d import RealAxisMigdal
from migdal_2d import Migdal
import shutil
import sys
from matplotlib.pyplot import *
from scipy import interpolate
from interpolator import Interp
import matplotlib.pyplot as plt
from a2F import corrected_a2F, corrected_lamb_mass, corrected_lamb_bare, corrected_a2F_imag
import matplotlib.colors as mcolors
from scipy.integrate import trapz

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

gr1 = [224./255,224./255,244./255]
b0 = [0.2,0.2,1.0]
g1 = [80./255,185./255,90./255]
#g1 = [80./255,0.8,90./255]

y1 = [1.0,1.0,81./255]
br1 =[120./255,81./255,0]
#br1 = [0.7,0.2,0.3]
w1 = [1,1,1]

c = mcolors.ColorConverter().to_rgb
rvb = make_colormap(
    [w1, gr1, 0.04, gr1, b0, 0.08,
     b0, g1, 0.15,  g1, y1, 0.35,
     y1, br1, 0.55, br1, w1])

plt.register_cmap(cmap=rvb)
mycmap = plt.get_cmap('CustomMap')

#c = mcolors.ColorConverter().to_rgb
#rvb = make_colormap(
#    [w1, gr1, 0.04, gr1, b0, 0.09,
#     b0, y1, 0.4,
#     y1, br1, 0.7, br1, w1])

def read_params(basedir, folder):
    
    params = {}
    for fname in os.listdir(os.path.join(basedir, 'data', folder)):
        if '.npy' in fname:
            data = np.load(os.path.join(basedir, 'data', folder, fname), allow_pickle=True)
            params[fname[:-4]] = data

    floats = ['beta', 'dens', 'dw', 'g0', 'mu', 'omega', \
              'idelta', 't', 'tp', 'wmax', 'wmin']
    ints   = ['dim', 'nk', 'nw']
    bools  = ['sc']
    
    for f in floats:
        params[f] = params[f][0]
    for i in ints:
        params[i] = int(params[i][0])
    for b in bools:
        params[b] = bool(params[b][0])
        
    params['band'] = band_square_lattice

    return params


    
def smoothedA(basedir, folder):

    #data = os.listdir(basedir+'data/'+folder)
    SR = np.load(basedir + 'data/' + folder + '/SR.npy')

    #params = setup_params(renormalized=True, beta=None, lamb=0.0)
    params = read_params(basedir, folder)
    
    beta = np.load(basedir+'data/' + folder + '/beta.npy')[0]
    params['beta'] = beta
    mu = np.load(basedir+'data/' + folder + '/mu.npy')[0]
    nk = np.load(basedir+'data/' + folder + '/nk.npy')[0]
    params['nk'] = nk
    nr = np.load(basedir+'data/' + folder + '/nw.npy')[0]
    params['nr'] = nr


    migdal = RealAxisMigdal(params, basedir)

    wn, vn, ek, w, nB, nF, DRbareinv = migdal.setup_realaxis()

    print('folder', folder)
    print('wtf len(w)', len(w))

    idxkf = np.argmin(np.abs(ek[:,0]))
    print('ik kF', idxkf)

    Nk = 1000

    ks = np.arange(-np.pi, np.pi, 2*np.pi/migdal.nk)
    #kxs = np.linspace(-np.pi, np.pi - 2*np.pi/migdal.nk, Nk)
    kxs = np.arange(-np.pi, np.pi, 2*np.pi/Nk)
    kys = ks[migdal.nk//4] * np.ones(Nk)

    if params['sc']:
        SRint = np.zeros((Nk, 1, migdal.nr, 2, 2), dtype=complex)
    else:
        SRint = np.zeros((Nk, 1, migdal.nr), dtype=complex)
        
    ekint = -2.0*migdal.t*(np.cos(kxs)+np.cos(kys)) - 4.0*migdal.tp*np.cos(kxs)*np.cos(kys)
    ekint = np.reshape(ekint, [-1,1])

    ks = np.concatenate((ks, [np.pi]))
    SR = np.concatenate((SR, SR[0,...][None,:,...]), axis=0)

    print('shpe nr', migdal.nr)
    print('shape SR', np.shape(SR))

    if params['sc']:
        for iw in range(migdal.nr):
            for a in range(2):
                for b in range(2):
                    '''
                    tckr = interpolate.splrep(ks, SR[:,migdal.nk//4,iw,a,b].real, s=0)
                    tcki = interpolate.splrep(ks, SR[:,migdal.nk//4,iw,a,b].imag, s=0)
                    SRint[:,0,iw,a,b] = interpolate.splev(kxs, tckr) + 1j*interpolate.splev(kxs, tcki)
                    '''
                    #I = interpolate.interp1d(ks, SR[:,migdal.nk//4,iw,a,b], kind='linear')                    
                    #SRint[:,0,iw,a,b] = I(kxs)
                    pass

    else:
         for iw in range(migdal.nr):
             Ir = interpolate.interp1d(ks, SR[:,migdal.nk//4,iw].real, kind='linear')
             Ii = interpolate.interp1d(ks, SR[:,migdal.nk//4,iw].imag, kind='linear')
             SRint[:,0,iw] = Ir(kxs) + 1j * Ii(kxs)

        
    GRint = migdal.compute_GR(w, ekint, mu, SRint) 

    print('shape GRint', np.shape(GRint))
    if params['sc']:
        return idxkf, -1.0/np.pi*GRint[:,0,:,0,0].imag
    return idxkf, -1.0/np.pi*GRint[:,0,:].imag


def smoothedB(basedir, folder):

    #data = os.listdir(basedir+'data/'+folder)
    PIR = np.load(basedir + 'data/' + folder + '/PIR.npy')

    #params = setup_params(renormalized=True, beta=None, lamb=0.0)
    params = read_params(basedir, folder)
    
    beta = np.load(basedir+'data/' + folder + '/beta.npy')[0]
    params['beta'] = beta
    mu = np.load(basedir+'data/' + folder + '/mu.npy')[0]
    nk = np.load(basedir+'data/' + folder + '/nk.npy')[0]
    params['nk'] = nk

    migdal = RealAxisMigdal(params, basedir)

    wn, vn, ek, w, nB, nF, DRbareinv = migdal.setup_realaxis()

    print('len(w)', len(w))
    

    Nk = 150

    ks = np.arange(-np.pi, np.pi, 2*np.pi/migdal.nk)

    # build path for interpolation

    kxs = np.linspace(-np.pi, np.pi - 2*np.pi/migdal.nk, Nk)
    kys = ks[migdal.nk//4] * np.ones(Nk)
    ekint = -2.0*migdal.t*(np.cos(kxs)+np.cos(kys)) - 4.0*migdal.tp*np.cos(kxs)*np.cos(kys)
    ekint = np.reshape(ekint, [-1,1])        

    def eval(Ir, Ii, kxs, kys):
        PIint = np.array([Ir(kx, ky) + 1j*Ii(kx, ky) for kx, ky in zip(kxs, kys)])
        #PIint = np.reshape(PIint, [-1,1,1])
        #ekint = -2.0*migdal.t*(np.cos(kxs)+np.cos(kys)) - 4.0*migdal.tp*np.cos(kxs)*np.cos(kys)
        #ekint = np.reshape(ekint, [-1,1])
        return np.reshape(PIint, [-1])

    PIint = np.zeros((3*Nk+1, 1, migdal.nr), dtype=complex)
    for iw in range(migdal.nr):
        Ir = interpolate.interp2d(ks, ks, PIR[:,:,iw].real, kind='linear')
        Ii = interpolate.interp2d(ks, ks, PIR[:,:,iw].imag, kind='linear')

        # construct the ys along the path
        kxs = np.arange(-np.pi, 0, np.pi/Nk)
        kys = kxs
        P1 = eval(Ir, Ii, kxs, kys)

        kxs = np.arange(0, -np.pi, -np.pi/Nk)
        kys = np.zeros(len(kxs))
        P2 = eval(Ir, Ii, kxs, kys)

        kys = np.arange(0, -np.pi, -np.pi/Nk)
        kxs = -np.pi*np.ones(len(kys))
        P3 = eval(Ir, Ii, kxs, kys)

        P4 = eval(Ir, Ii, [-np.pi], [-np.pi])
    
        PIint[:,0,iw] = np.concatenate((P1, P2, P3, P4))

    print('end point diff', np.mean(np.abs(PIint[0,0,:]-PIint[-1,0,:])))
    print('end point -1 diff', np.mean(np.abs(PIint[0,0,:]-PIint[-2,0,:])))
            
    DRint = migdal.compute_DR(DRbareinv, PIint)    

    print('shape DRint', np.shape(DRint))
    return -1.0/np.pi * DRint[:,0,:].imag



def debug():
    #basedir = '../debug/'
    
    basedir = '../'
    folder0 = 'data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'    
    params = read_params(basedir, folder0)
    
    DR = np.load(basedir + 'data/' + folder0 + '/DR.npy')
    B = -1.0/np.pi * DR.imag
    nr = np.shape(B)[2]
    Bn = np.sum(B[:,:,nr//2:], axis=2)
    print('norm check DR renorm ', np.mean(Bn), ' +- ', np.std(Bn))
    
    
    basedir = '../'
    folder1 = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'    
    params = read_params(basedir, folder1)    
    
    print('dw', params['dw'])
    print('idelta', params['idelta'])

    DR = np.load(basedir + 'data/' + folder1 + '/DR.npy')
    B = -1.0/np.pi * DR.imag
    nr = np.shape(B)[2]
    Bn = np.sum(B[:,:,nr//2:], axis=2)
    print('norm check DR unrenorm ', np.mean(Bn), ' +- ', np.std(Bn))
    
    return


    print('folder', folder1)

    def imag():
        lamba2Fiku, _ = corrected_a2F_imag(basedir, folder1, ntheta=40)

    imag()        
    lamba2Fiku = np.load(basedir + 'data/' + folder1 + '/lambk_a2F_imag.npy')

    return




# a2F
def compute_a2F():
    #basedir = '/home/groups/tpd/bln/data/elph/imagaxis/2d_sc/'
 
    #folders = sorted(os.listdir(basedir+'data/'))
    
    #basedir = '../'
    basedir = '/scratch/users/bln/elph/data/2d/'

    #folder0 = 'data_renormalized_nk96_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    #folder1 = 'data_unrenormalized_nk96_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    
    folder0 = 'data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    folder1 = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    
    
    params = read_params(basedir, folder0)
    print('dw', params['dw'])
    print('idelta', params['idelta'])

    params = read_params(basedir, folder1)
    print('dw', params['dw'])
    print('idelta', params['idelta'])
    
    
    for key in params:
        print(key, type(params[key]))

    print('folder', folder1)


    def main_computation():
        # main computation
        print('renormalized lamb')
        _, lambbarekr, weightsr = corrected_lamb_bare(basedir, folder0, ntheta=80)
        lambmasskr, _ = corrected_lamb_mass(basedir, folder0, ntheta=80)
        lamba2Fkr, _ = corrected_a2F(basedir, folder0, ntheta=40)
        
        np.save(basedir + 'lambbarekr', lambbarekr)
        np.save(basedir + 'lambmasskr', lambmasskr)
        np.save(basedir + 'lamba2Fkr', lamba2Fkr)
        np.save(basedir + 'weightsr', weightsr)

        
        print('unrenormalized lamb')
        _, lambbareku, weightsu = corrected_lamb_bare(basedir, folder1, ntheta=80)
        lambmassku, _ = corrected_lamb_mass(basedir, folder1, ntheta=80)
        lamba2Fku, _ = corrected_a2F(basedir, folder1, ntheta=40)

        np.save(basedir + 'lambbareku', lambbareku)
        np.save(basedir + 'lambmassku', lambmassku)
        np.save(basedir + 'lamba2Fku', lamba2Fku)
        np.save(basedir + 'weightsu', weightsu)
    
    #############################
    # compute a2F
    main_computation()    
    

    # compute lamb_a2F using imaginary axis only
    def imag():
        lamba2Fikr, _ = corrected_a2F_imag(basedir, folder0, ntheta=40)
        lamba2Fiku, _ = corrected_a2F_imag(basedir, folder1, ntheta=40)

    imag()        
    
    lamba2Fikr = np.load(basedir + 'data/' + folder0 + '/lambk_a2F_imag.npy')
    lamba2Fiku = np.load(basedir + 'data/' + folder1 + '/lambk_a2F_imag.npy')

    
    lambbarekr = np.load(basedir + 'lambbarekr.npy')
    lambmasskr = np.load(basedir + 'lambmasskr.npy')
    lamba2Fkr  = np.load(basedir + 'lamba2Fkr.npy')
    lamba2Fkri = np.load(basedir + 'data/' + folder0 + '/lambk_a2F_imag.npy')
    lambbareku = np.load(basedir + 'lambbareku.npy')
    lambmassku = np.load(basedir + 'lambmassku.npy')
    lamba2Fku  = np.load(basedir + 'lamba2Fku.npy')
    weightsu   = np.load(basedir + 'weightsu.npy')
    weightsr   = np.load(basedir + 'weightsr.npy')

    
    
    f = plt.figure()
    ntheta = len(lambmasskr)//4
    dtheta = np.pi / (2 * ntheta)
    thetas = np.arange(dtheta/2, np.pi/2, dtheta)
    print('ntheta', ntheta, 'len thetas', len(thetas))
    plt.plot(thetas * 180/np.pi, lambmasskr[:ntheta], 'k')
    
    ntheta = len(lamba2Fkr)
    dtheta = np.pi / (2 * ntheta)
    thetas = np.arange(dtheta/2, np.pi/2, dtheta)
    plt.plot(thetas * 180/np.pi, lamba2Fkr, 'k', linestyle='dotted')
    #plt.plot(thetas * 180/np.pi, lamba2Fkri * 2 / 2.5, 'k', linestyle='dotted')
    plt.ylabel('$\lambda(\mathbf{k})$', fontsize=14)
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.legend(['$\lambda_{m}$', '$\lambda_{a2F}$'], fontsize=12)
    plt.xlim(0, 90)
    
    ax = f.add_axes([0.17, 0.4, 0.24, 0.24])
    thetas = np.linspace(0, np.pi/2, 100)
    ax.plot(np.pi - 0.8*np.pi*np.cos(thetas), np.pi - 0.8*np.pi*np.sin(thetas), 'k-')
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, np.pi)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.plot([0, np.pi], [np.pi/2, np.pi], 'r')
    ax.annotate(r'$\theta$', xy=(0.45, 0.83), xycoords='axes fraction')
    ax.set_aspect(1)
    
    plt.savefig(basedir+'lambkr')
    plt.close()
    
    
    
    plt.figure()
    #plt.plot(np.linspace(0, 1, len(lambbareku)//4), lambbareku[:len(lambbareku)//4], 'k--')
    plt.plot(np.linspace(0, 1, len(lambmassku)//4), lambmassku[:len(lambmassku)//4], 'k')
    plt.plot(np.linspace(0, 1, len(lamba2Fku)), lamba2Fku, 'k', linestyle='dotted')
    plt.savefig(basedir+'lambku')
    plt.close()
    
    plt.figure()
    plt.plot(weightsr)
    plt.plot(weightsu)
    plt.savefig(basedir+'weights')
    plt.close()

    
    w0   = np.load(basedir+'data/'+folder0+'/w.npy')
    a2F0 = np.load(basedir+'data/'+folder0+'/a2F.npy')
    lamb0 = np.load(basedir+'data/'+folder0+'/lamb_from_a2F.npy')[0]
    #lamb0el = np.load(basedir+'data/'+folder0+'/lamb_electronic.npy')[0]
    w1   = np.load(basedir+'data/'+folder1+'/w.npy')
    a2F1 = np.load(basedir+'data/'+folder1+'/a2F.npy')
    lamb1 = np.load(basedir+'data/'+folder1+'/lamb_from_a2F.npy')[0]
    #lamb1el = np.load(basedir+'data/'+folder1+'/lamb_electronic.npy')[0]
    
    lamb0mass = np.load(basedir+'data/'+folder0+'/lamb_mass.npy')[0]
    lamb1mass = np.load(basedir+'data/'+folder1+'/lamb_mass.npy')[0]
    
    print('len(w0)', len(w0), 'len(w1)', len(w1))
    
    figure()
    plot(w1, a2F1, '.-')
    plot(w0, a2F0, '.-')
    legend(['$\lambda_{ME}$'+'={:.2f}'.format(lamb1), '$\lambda_{RME}$'+'={:.2f}'.format(lamb0)])
    xlim(0, 0.3)
    #ylim(0, 0.8)
    ylim(0, 3.1)
    xlabel('$\omega / t$', fontsize=13)
    ylabel(r'$\alpha^2 F$', fontsize=13)
    savefig(basedir+'a2Fcomp')
    close()
    
    print('lamb a2F unrenorm : ', lamb1)
    #print('lamb bare unrenorm : ', lamb1el)
    print('lamb mass unrenorm : ', lamb1mass)
    print('lamb a2F renorm : ', lamb0)
    #print('lamb electronic renorm : ', lamb0el)
    print('lamb mass renorm : ', lamb0mass)


    # McMillan Tc formula from a2F
    
    print('simple check of integral')
    print('lamb0', lamb0)
    dw = (w0[-1]-w0[0]) / (len(w0)-1)
    nr = len(a2F0)
    lambcheck = 2 * np.sum(a2F0[nr//2+1:] / w0[nr//2+1:]) * dw
    print('lambcheck', lambcheck)
    lambcheckquad = 2 * np.sum(a2F0[nr//2+1:] / w0[nr//2+1:]) * dw
    L = len(w0[nr//2+1:])
    ws =  np.arange(0, dw*L, dw)
    print('lambcheck using quad : ', 2 * trapz(a2F0[nr//2+1:] / w0[nr//2+1:], ws))
    
    print('normalization a2F renormalized : ', trapz(a2F0[nr//2+1:], ws))
    print('normalization a2F unrenormalized : ', trapz(a2F1[nr//2+1:], ws))
    
    
    wln = np.exp( 2 / lamb0 * np.sum(np.log(w0[nr//2+1:]) * a2F0[nr//2+1:] / w0[nr//2+1:]) * dw)
    print('wln ', wln)
    
    print('Tc ', wln/1.2 * np.exp(-1.04*(1 + lamb0) / lamb0))


# plotting
def old_plotting():
    
    basedir = '../'
    
    folder0 = 'data_renormalized_nk96_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    folder1 = 'data_unrenormalized_nk96_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    
    #folder0 = 'data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    #folder1 = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    
    
    params = read_params(basedir, folder0)
    for key in params:
        print(key, type(params[key]))
        
    # exit()
    # print(key, np.shape(params[key]))
    
    print(folder0)
    print(folder1)

    #avg_omega(basedir, folder0)
    
    f = figure()
    f.set_size_inches(20, 5*20/25)

    #f = figure()


    fx, fy = 0.05, 0.9
    fs = 12

    sw = 0.16
    sh = 0.8
    ax = f.add_axes([0.05, 0.1, sw, sh])
    wr = np.load(basedir+'data/'+folder1+'/w.npy')
    GR = np.load(basedir+'data/'+folder1+'/GR.npy')
    nk = np.load(basedir+'data/'+folder1+'/nk.npy')[0]
    idxkf, A = smoothedA(basedir, folder1)    
    #A  = -1.0/np.pi * GR[:,nk//4,:,0,0].imag
    #ax.plot(wr, A)
    im = ax.imshow(np.log(1+A.T)/np.log(4), origin='lower', extent=[-np.pi, np.pi, wr[0], wr[-1]], cmap='gnuplot', aspect=0.6) 
    #colorbar(im, ax=ax)
    ax.set_xlim(0, 2.8)
    ax.set_ylim(-1.5, 2.1)
    ax.set_ylabel('$\omega/t$')
    ax.set_xlabel('k/a')
    ax.annotate('A', xy=(fx, fy), xycoords='axes fraction', fontsize=fs, color='w')

    #f = figure()
    ax = f.add_axes([0.24, 0.1, sw, sh])
    wr = np.load(basedir+'data/'+folder0+'/w.npy')
    GR = np.load(basedir+'data/'+folder0+'/GR.npy')
    nk = np.load(basedir+'data/'+folder0+'/nk.npy')[0]
    idxkf, A = smoothedA(basedir, folder0)    
    #A  = -1.0/np.pi * GR[:,nk//4,:,0,0].imag
    #ax.plot(wr, A[nk//4, nk//4])
    im = ax.imshow(np.log(1+A.T)/np.log(4), origin='lower', extent=[-np.pi, np.pi, wr[0], wr[-1]], cmap='gnuplot', aspect=0.6) 
    #colorbar(im, ax=ax)
    ax.set_xlim(0, 2.8)
    ax.set_ylim(-1.5, 2.1)
    #ax.set_ylabel('$\omega/t$')
    ax.set_xlabel('k/a')
    ax.annotate('B', xy=(fx, fy), xycoords='axes fraction', fontsize=fs, color='w')


    #f = figure()
    ax = f.add_axes([0.43, 0.1, sw, sh])
    wr = np.load(basedir+'data/'+folder0+'/w.npy')
    nk = np.load(basedir+'data/'+folder0+'/nk.npy')[0]
    #DR = np.load(basedir+'data/'+folder0+'/DR.npy')
    #B = -1.0/np.pi * DR[:,nk//4,:].imag
    B = smoothedB(basedir, folder0)
    #im = ax.imshow(np.log(1+np.abs(B.T))/np.log(4), origin='lower', extent=[-np.pi, np.pi, wr[0], wr[-1]], cmap='gnuplot', aspect=1.2, vmin=0)
    im = ax.imshow(B.T, origin='lower', extent=[-np.pi, np.pi, wr[0], wr[-1]], cmap='gnuplot', aspect='auto', vmin=0)
    ax.hlines(1.0, -np.pi, np.pi, linestyle='--', color='w')
    #colorbar(im, ax=ax)
    ax.set_xlim(0, np.pi)
    ax.set_xticks((0, np.pi/3, 2*np.pi/3, np.pi))
    ax.set_xticklabels(('($\pi$,$\pi$)', '(0,0)', '($\pi$,0)', '($\pi$,$\pi$)'))
    ax.set_ylim(0, 0.2)
    #ax.set_ylabel('phonon spectral function')
    #ax.set_ylabel('$\omega/t$')
    ax.set_xlabel('k/a')
    ax.annotate('C', xy=(fx, fy), xycoords='axes fraction', fontsize=fs, color='w')


    #f = figure()
    ax = f.add_axes([0.62, 0.2, 0.16, 0.6])
    wr = np.load(basedir+'data/'+folder1+'/w.npy')
    nk = np.load(basedir+'data/'+folder1+'/nk.npy')[0]
    SR = np.load(basedir+'data/'+folder1+'/SR.npy')
    
    if params['sc']:
        ax.plot(wr, SR.real[idxkf,0,:,0,0])
        ax.plot(wr, SR.imag[idxkf,0,:,0,0])
    else:
        ax.plot(wr, SR.real[idxkf,0,:])
        ax.plot(wr, SR.imag[idxkf,0,:])

    wr = np.load(basedir+'data/'+folder0+'/w.npy')
    nk = np.load(basedir+'data/'+folder0+'/nk.npy')[0]
    SR = np.load(basedir+'data/'+folder0+'/SR.npy')
    if params['sc']:
        ax.plot(wr, SR.real[idxkf,0,:,0,0], color='C0', linestyle='--')
        ax.plot(wr, SR.imag[idxkf,0,:,0,0], color='C1', linestyle='--')
    else:
        ax.plot(wr, SR.real[idxkf,0,:], color='C0', linestyle='--')
        ax.plot(wr, SR.imag[idxkf,0,:], color='C1', linestyle='--')
    
    #ax.set_ylabel('selfenergy at antinodal kF')
    ax.set_xlabel('$\omega/t$')
    ax.annotate('D', xy=(fx, fy), xycoords='axes fraction', fontsize=fs)


    #f = figure()
    ax = f.add_axes([0.81, 0.2, 0.16, 0.6])
    wr = np.load(basedir+'data/'+folder0+'/w.npy')
    nk = np.load(basedir+'data/'+folder0+'/nk.npy')[0]
    SR = np.load(basedir+'data/'+folder0+'/SR.npy')
    if params['sc']:
        ax.plot(wr, SR.real[idxkf,0,:,0,1])
        ax.plot(wr, SR.imag[idxkf,0,:,0,1])
    else:
        ax.plot(wr, SR.real[idxkf,0,:])
        ax.plot(wr, SR.imag[idxkf,0,:])
    #ax.set_ylabel('anomalous selfenergy at antinodal kF')
    ax.set_xlabel('$\omega/t$')
    ax.annotate('E', xy=(fx, fy), xycoords='axes fraction', fontsize=fs)

    
    savefig(basedir+'test')
    close()



def figure1_parts():  
    basedir = '../'
    
    #rfolder = 'data_renormalized_nk96_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    #ufolder = 'data_unrenormalized_nk96_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    
    rfolder = 'data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    ufolder = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    
    
    params = read_params(basedir, ufolder)
    nk = params['nk']
    nr = len(params['w'])
    uGR = np.load(os.path.join(basedir, 'data', ufolder, 'GR.npy'))
    rGR = np.load(os.path.join(basedir, 'data', rfolder, 'GR.npy'))
    
    w = np.load(os.path.join(basedir, 'data', rfolder, 'w.npy'))
    dw = np.load(os.path.join(basedir, 'data', rfolder, 'w.npy'))[0]
    
    # BZ path for B(k)
    rDR = np.load(os.path.join(basedir, 'data', rfolder, 'DR.npy'))
    
    
    rB = []
    for i in range(nk//2):
        rB.append(rDR[i,i].imag)
    for i in range(nk//2):
        rB.append(rDR[nk//2-i, nk//2].imag)
    for i in range(nk//2):
        rB.append(rDR[0, nk//2-i].imag)
    rB.append(rDR[0,0].imag)
    rB = -1.0/np.pi * np.array(rB)
    
    
    uA = -1.0/np.pi * uGR[:,nk//4].imag
    rA = -1.0/np.pi * rGR[:,nk//4].imag
    
    plt.figure()
    plt.imshow(uA.T, origin='lower', aspect='auto', interpolation='bilinear',
               extent=[-np.pi,np.pi,params['wmin'],params['wmax']])
    plt.xlim(0, 2)
    plt.ylim(-1.5, 1.5)
    plt.savefig(basedir+'uA')
    
    
    plt.figure()
    plt.imshow(rA.T, origin='lower', aspect='auto', interpolation='bilinear',
               extent=[-np.pi,np.pi,params['wmin'],params['wmax']])
    plt.xlim(0, 2)
    plt.ylim(-1.5, 1.5)
    plt.savefig(basedir+'rA')
    
    
    plt.figure()
    plt.imshow(rB.T, origin='lower', aspect='auto', vmin=0, interpolation='bilinear',
               extent=[-np.pi,np.pi,params['wmin'],params['wmax']], cmap='Greys')
    plt.ylim(0, 0.17)
    plt.savefig(basedir+'f1B')
    plt.close()
    
    '''
    plt.figure()
    B = smoothedB(basedir, rfolder)
    plt.imshow(B.T, origin='lower', extent=[-np.pi, np.pi, params['wmin'], params['wmax']],
               aspect='auto', vmin=0, interpolation='bilinear')
    #ax.hlines(1.0, -np.pi, np.pi, linestyle='--', color='w')
    plt.ylim(0, 0.2)
    plt.savefig(basedir+'B2')
    plt.close()
    '''
    
    PIR = np.load(os.path.join(basedir, 'data', rfolder, 'PIR.npy'))
    plt.figure()
    plt.plot(w, PIR[0,0].real, 'k')
    plt.plot(w, PIR[0,0].imag, 'k', linestyle='--')
    plt.plot(w, PIR[nk//2,nk//2].real, 'k')
    plt.plot(w, PIR[nk//2,nk//2].imag, 'k', linestyle='--')
    plt.savefig(basedir+'f1PIR')
    plt.close()    
    
    
    rSR = np.load(os.path.join(basedir, 'data', rfolder, 'SR.npy'))
    uSR = np.load(os.path.join(basedir, 'data', ufolder, 'SR.npy'))
    
    '''
    for ik in range(0, nk//4, 4):
        
        # find k index where maximal A is close to 0
        uA = -1.0/np.pi * uGR[:, ik].imag
        rA = -1.0/np.pi * rGR[:, ik].imag
        
        uwidxs = np.argmax(uA, axis=1)
        rwidxs = np.argmax(rA, axis=1)
        
        
        uidx = np.argmin(np.abs(uwidxs - nr//2))
        ridx = np.argmin(np.abs(rwidxs - nr//2))
        
        plt.figure()
        plt.plot(w, uA[uidx])
        plt.plot(w, rA[ridx])
        plt.savefig(basedir+'A%d'%ik)
        
        
        plt.figure()
        plt.title('%d %d'%(uidx, ridx))
        plt.plot(w, uSR[uidx, ik].real)
        plt.plot(w, rSR[ridx, ik].real)
        plt.savefig(basedir+'SR%d'%ik)
    '''  
    
    w0   = np.load(basedir+'data/'+ufolder+'/w.npy')
    ua2F = np.load(basedir+'data/'+ufolder+'/a2F.npy')
    ulamb = np.load(basedir+'data/'+ufolder+'/lamb_from_a2F.npy')[0]
    ulambel = np.load(basedir+'data/'+ufolder+'/lamb_electronic.npy')[0]
    w1   = np.load(basedir+'data/'+rfolder+'/w.npy')
    ra2F = np.load(basedir+'data/'+rfolder+'/a2F.npy')
    rlamb = np.load(basedir+'data/'+rfolder+'/lamb_from_a2F.npy')[0]
    rlambel = np.load(basedir+'data/'+rfolder+'/lamb_electronic.npy')[0]
    
    figure()
    plot(w1, ua2F)
    plot(w0, ra2F)
    legend(['$\lambda_{ME}$'+'={:.2f}'.format(ulamb), '$\lambda_{RME}$'+'={:.2f}'.format(rlamb)])
    xlim(0, 0.3)
    ylim(0, 1.75)
    xlabel('$\omega / t$', fontsize=13)
    ylabel(r'$\alpha^2 F$', fontsize=13)
    savefig(basedir+'f1a2F')
    
    #ax = f.add_axes([0.1, 0.55, 0.33, 0.33])
    #ax.imshow(rB.T, origin='lower', extent=[-np.pi,np.pi,
    #                params['wmin'],params['wmax']],aspect='auto',
    #                vmin=0)
    #ax.set_ylim(0, 0.55)
    
    
def figure1():  
    basedir = '../'
    
    rfolder = 'data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    ufolder = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
      
    params = read_params(basedir, ufolder)
    nk = params['nk']
    nr = len(params['w'])
    
    w = np.load(os.path.join(basedir, 'data', rfolder, 'w.npy'))
    dw = np.load(os.path.join(basedir, 'data', rfolder, 'w.npy'))[0]
    
    # BZ path for B(k)
    rDR = np.load(os.path.join(basedir, 'data', rfolder, 'DR.npy'))
        
    rB = []
    for i in range(nk//2):
        rB.append(rDR[i,i].imag)
    for i in range(nk//2):
        rB.append(rDR[nk//2-i, nk//2].imag)
    for i in range(nk//2):
        rB.append(rDR[0, nk//2-i].imag)
    rB.append(rDR[0,0].imag)
    rB = -1.0/np.pi * np.array(rB)  
    
    f = plt.figure()
    f.set_size_inches(7, 15)
    
    fsz = 13
    
    ax = f.add_axes([0.14, 0.69, 0.8, 0.27])
    ax.imshow(rB.T, origin='lower', aspect='auto', vmin=0, interpolation='bilinear',
               extent=[-np.pi,np.pi,params['wmin'],params['wmax']], cmap='Greys')
    ax.set_ylim(0, 0.17)
    ax.set_ylabel('$\omega/t$', fontsize=fsz)
    ax.set_xticks((-np.pi, -np.pi/3, np.pi/3, np.pi))                                                                                    
    ax.set_xticklabels(('($\pi$,$\pi$)', '(0,0)', '($\pi$,0)', '($\pi$,$\pi$)'), fontsize=fsz)                                                     
    
    
    ax = f.add_axes([0.14, 0.38, 0.8, 0.27])
    PIR = np.load(os.path.join(basedir, 'data', rfolder, 'PIR.npy'))
    ax.plot(w, PIR[0,0].real, 'C0')
    ax.plot(w, PIR[0,0].imag, 'C0', linestyle='--')
    ax.plot(w, PIR[nk//2,nk//2].real, 'C1')
    ax.plot(w, PIR[nk//2,nk//2].imag, 'C1', linestyle='--')
    ax.legend(['Re $\Pi(\pi,\pi)$', 'Im $\Pi(\pi,\pi)$',
               'Re $\Pi(0,0)$', 'Re $\Pi(0,0)$'])
    ax.set_ylabel('$\omega/t$', fontsize=fsz)
    ax.set_xlabel('$\omega/t$', fontsize=fsz)
    
    
    ax = f.add_axes([0.14, 0.07, 0.8, 0.27])
    w0   = np.load(basedir+'data/'+ufolder+'/w.npy')
    ua2F = np.load(basedir+'data/'+ufolder+'/a2F.npy')
    ulamb = np.load(basedir+'data/'+ufolder+'/lamb_from_a2F.npy')[0]
    ulambel = np.load(basedir+'data/'+ufolder+'/lamb_electronic.npy')[0]
    w1   = np.load(basedir+'data/'+rfolder+'/w.npy')
    ra2F = np.load(basedir+'data/'+rfolder+'/a2F.npy')
    rlamb = np.load(basedir+'data/'+rfolder+'/lamb_from_a2F.npy')[0]
    rlambel = np.load(basedir+'data/'+rfolder+'/lamb_electronic.npy')[0]
    
    ax.plot(w1, ua2F)
    ax.plot(w0, ra2F)
    ax.legend(['$\lambda_{ME}$'+'={:.2f}'.format(ulamb), '$\lambda_{RME}$'+'={:.2f}'.format(rlamb)])
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 3.1)
    ax.set_xlabel('$\omega / t$', fontsize=fsz)
    ax.set_ylabel(r'$\alpha^2 F$', fontsize=fsz)
    
    plt.savefig(basedir+'f1')




def figure2():  
    basedir = '../'
    
    rfolder = 'data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    ufolder = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
        
    params = read_params(basedir, ufolder)
    nk = params['nk']
    nr = len(params['w'])
    uGR = np.load(os.path.join(basedir, 'data', ufolder, 'GR.npy'))
    rGR = np.load(os.path.join(basedir, 'data', rfolder, 'GR.npy'))
    
    w = np.load(os.path.join(basedir, 'data', rfolder, 'w.npy'))
    dw = np.load(os.path.join(basedir, 'data', rfolder, 'w.npy'))[0]
        
    uA = -1.0/np.pi * uGR[:,nk//4].imag
    rA = -1.0/np.pi * rGR[:,nk//4].imag
    
    _, suA = smoothedA(basedir, ufolder)
    _, srA = smoothedA(basedir, rfolder)
    
    fsz = 13
    f = plt.figure()
    f.set_size_inches([6, 20])
    
    #cmap = mycmap
    #cmap = 'afmhot_r'
    cmap = 'Greys'
    #cmap = 'gnuplot'
    #cmap = 'viridis'
    

    ax = f.add_axes([0.14, 0.74, 0.8, 0.2])
    ax.imshow(suA.T, origin='lower', aspect='auto', interpolation='bilinear',
               extent=[-np.pi,np.pi,params['wmin'],params['wmax']], cmap=cmap)
    ax.set_ylabel('$\omega/t$', fontsize=fsz)
    ax.set_xlabel('$k/a$', fontsize=fsz)
    ax.set_xlim(0, 1.75)
    ax.set_ylim(-1.5, 1.5)
    
    ax = f.add_axes([0.14, 0.51, 0.8, 0.2])
    plt.imshow(srA.T, origin='lower', aspect='auto', interpolation='bilinear',
               extent=[-np.pi,np.pi,params['wmin'],params['wmax']], cmap=cmap)
    ax.set_ylabel('$\omega/t$', fontsize=fsz)
    ax.set_xlabel('$k/a$', fontsize=fsz)
    ax.set_xlim(0, 1.75)
    ax.set_ylim(-1.5, 1.5)
    
    ax = f.add_axes([0.14, 0.28, 0.8, 0.2])
    uSR = np.load(os.path.join(basedir, 'data', ufolder, 'SR.npy'))
    rSR = np.load(os.path.join(basedir, 'data', rfolder, 'SR.npy'))
    uA = -1.0/np.pi * uGR[:,0].imag
    rA = -1.0/np.pi * rGR[:,0].imag
    uwidxs = np.argmax(uA, axis=1)
    uidx = np.argmin(np.abs(uwidxs - nr//2))
    rwidxs = np.argmax(rA, axis=1)
    ridx = np.argmin(np.abs(rwidxs - nr//2))
    
    ax.plot(w, uSR[uidx,0].real, 'C0')
    ax.plot(w, uSR[uidx,0].imag, 'C0', linestyle='--')
    ax.plot(w, rSR[ridx,0].real, 'C1')
    ax.plot(w, rSR[ridx,0].imag, 'C1', linestyle='--')
    ax.legend(['Re $\Sigma_{ME}$', 'Im $\Sigma_{ME}$',
               'Re $\Sigma_{RME}$', 'Im $\Sigma_{RME}$'], fontsize=fsz)
    ax.set_ylabel('$\omega/t$', fontsize=fsz)
    ax.set_xlabel('$\omega/t$', fontsize=fsz)
    ax.set_xlim(-3, 3)
    
    ax = f.add_axes([0.14, 0.05, 0.8, 0.2])
    ax.plot(w, uA[uidx], 'C0')
    ax.plot(w, rA[ridx], 'C1')
    ax.legend(['$A_{ME}$', '$A_{RME}$'], fontsize=fsz)
    ax.set_ylabel('$\omega/t$', fontsize=fsz)
    ax.set_xlabel('$\omega/t$', fontsize=fsz)
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(0, 5)

    plt.savefig(basedir+'f2')
    plt.close()
    

def figure2_horizontal():  
    basedir = '../'
    
    rfolder = 'data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    ufolder = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
        
    params = read_params(basedir, ufolder)
    nk = params['nk']
    nr = len(params['w'])
    uGR = np.load(os.path.join(basedir, 'data', ufolder, 'GR.npy'))
    rGR = np.load(os.path.join(basedir, 'data', rfolder, 'GR.npy'))
    
    w = np.load(os.path.join(basedir, 'data', rfolder, 'w.npy'))
    dw = np.load(os.path.join(basedir, 'data', rfolder, 'w.npy'))[0]
        
    uA = -1.0/np.pi * uGR[:,nk//4].imag
    rA = -1.0/np.pi * rGR[:,nk//4].imag
    
    _, suA = smoothedA(basedir, ufolder)
    _, srA = smoothedA(basedir, rfolder)
    
    fsz = 17
    
    f = plt.figure()
    f.set_size_inches([25, 5])
    
    #cmap = mycmap
    #cmap = 'afmhot_r'
    cmap = 'Greys'
    #cmap = 'gnuplot'
    #cmap = 'viridis'
    

    ax = f.add_axes([0.05, 0.14, 0.19, 0.8])
    ax.imshow(suA.T, origin='lower', aspect='auto', interpolation='bilinear',
               extent=[-np.pi,np.pi,params['wmin'],params['wmax']], cmap=cmap)
    ax.set_ylabel('$\omega/t$', fontsize=fsz)
    ax.set_xlabel('$k/a$', fontsize=fsz)
    ax.set_xlim(0, 1.75)
    ax.set_ylim(-1.5, 1.5)
    
    ax = f.add_axes([0.28, 0.14, 0.19, 0.8])
    plt.imshow(srA.T, origin='lower', aspect='auto', interpolation='bilinear',
               extent=[-np.pi,np.pi,params['wmin'],params['wmax']], cmap=cmap)
    ax.set_ylabel('$\omega/t$', fontsize=fsz)
    ax.set_xlabel('$k/a$', fontsize=fsz)
    ax.set_xlim(0, 1.75)
    ax.set_ylim(-1.5, 1.5)
    
    ax = f.add_axes([0.51, 0.14, 0.19, 0.8])
    uSR = np.load(os.path.join(basedir, 'data', ufolder, 'SR.npy'))
    rSR = np.load(os.path.join(basedir, 'data', rfolder, 'SR.npy'))
    uA = -1.0/np.pi * uGR[:,0].imag
    rA = -1.0/np.pi * rGR[:,0].imag
    uwidxs = np.argmax(uA, axis=1)
    uidx = np.argmin(np.abs(uwidxs - nr//2))
    rwidxs = np.argmax(rA, axis=1)
    ridx = np.argmin(np.abs(rwidxs - nr//2))
    
    ax.plot(w, uSR[uidx,0].real, 'C0')
    ax.plot(w, uSR[uidx,0].imag, 'C0', linestyle='--')
    ax.plot(w, rSR[ridx,0].real, 'C1')
    ax.plot(w, rSR[ridx,0].imag, 'C1', linestyle='--')
    ax.legend(['Re $\Sigma_{ME}$', 'Im $\Sigma_{ME}$',
               'Re $\Sigma_{RME}$', 'Im $\Sigma_{RME}$'], fontsize=fsz)
    ax.set_ylabel('$\omega/t$', fontsize=fsz)
    ax.set_xlabel('$\omega/t$', fontsize=fsz)
    ax.set_xlim(-3, 3)
    
    ax = f.add_axes([0.74, 0.14, 0.19, 0.8])
    ax.plot(w, uA[uidx], 'C0')
    ax.plot(w, rA[ridx], 'C1')
    ax.legend(['$A_{ME}$', '$A_{RME}$'], fontsize=fsz)
    ax.set_ylabel('$\omega/t$', fontsize=fsz)
    ax.set_xlabel('$\omega/t$', fontsize=fsz)
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(0, 5)

    plt.savefig(basedir+'f2')
    plt.close()

    
#figure1()
#figure2_horizontal()

compute_a2F()

#debug()


#old_plotting()


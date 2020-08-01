
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
from functions import read_params


# a2F
def compute_a2F():
    #basedir = '/scratch/users/bln/elph/data/2dfixedn/'
    #basedir = '/scratch/users/bln/elph/data/single_iter/'
    #folder0 = 'data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    #folder1 = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'

    basedir = '/scratch/users/bln/elph/data/2dn0p786/'
    folder0 = 'data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.786_beta16.0000_QNone'
    folder1 = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.786_beta16.0000_QNone'
        

    params = read_params(basedir, folder0)
    print('renormalized:')
    print('dw', params['dw'])
    print('idelta', params['idelta'])

    params = read_params(basedir, folder1)
    print('unrenormalized')
    print('dw', params['dw'])
    print('idelta', params['idelta'])
    
    
    for key in params:
        print(key, type(params[key]))

    print('folder', folder1)

    def main_computation():
        # main computation
        print('\n\n\nrenormalized lamb')
        _, lambbarekr, weightsr = corrected_lamb_bare(basedir, folder0, ntheta=80)
        lambmasskr, _ = corrected_lamb_mass(basedir, folder0, ntheta=80)
        lamba2Fkr, _ = corrected_a2F(basedir, folder0, ntheta=80)
        
        np.save(basedir + 'lambbarekr', lambbarekr)
        np.save(basedir + 'lambmasskr', lambmasskr)
        np.save(basedir + 'lamba2Fkr', lamba2Fkr)
        np.save(basedir + 'weightsr', weightsr)

        
        print('\n\n\nunrenormalized lamb')
        _, lambbareku, weightsu = corrected_lamb_bare(basedir, folder1, ntheta=80)
        lambmassku, _ = corrected_lamb_mass(basedir, folder1, ntheta=80)
        lamba2Fku, _ = corrected_a2F(basedir, folder1, ntheta=80)

        np.save(basedir + 'lambbareku', lambbareku)
        np.save(basedir + 'lambmassku', lambmassku)
        np.save(basedir + 'lamba2Fku', lamba2Fku)
        np.save(basedir + 'weightsu', weightsu)

    
    #############################
    # compute a2F
    #main_computation()    

    # compute lamb_a2F using imaginary axis only
    def imag():
        lamba2Fikr, _ = corrected_a2F_imag(basedir, folder0, ntheta=80)
        lamba2Fiku, _ = corrected_a2F_imag(basedir, folder1, ntheta=80)

    #imag()        
    
    lamba2Fikr = np.load(basedir + 'data/' + folder0 + '/lambk_a2F_imag.npy')
    lamba2Fiku = np.load(basedir + 'data/' + folder1 + '/lambk_a2F_imag.npy')

    
    lambbarekr = np.load(basedir + 'lambbarekr.npy')
    lambmasskr = np.load(basedir + 'lambmasskr.npy')
    lamba2Fkr  = np.load(basedir + 'lamba2Fkr.npy')
    #lamba2Fkri = np.load(basedir + 'data/' + folder0 + '/lambk_a2F_imag.npy')
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
    plot(w1, a2F1)
    plot(w0, a2F0)
    legend(['$\lambda_{ME}$'+'=0.4'.format(lamb1), '$\lambda_{RME}$'+'=1.3'.format(lamb0)], fontsize=14, loc=2)
    xlim(0, 0.25)
    #ylim(0, 0.8)
    ylim(0, 2.0)
    xlabel('$\omega / t$', fontsize=13)
    ylabel(r'$\alpha^2 F$', fontsize=13)
    savefig(basedir+'a2Fcomp', transparent=True)
    close()

    dw0 = (w0[-1]-w0[0]) / (len(w0)-1)
    dw1 = (w1[-1]-w1[0]) / (len(w1)-1)
    
    print('lamb a2F unrenorm : ', lamb1)
    #print('lamb bare unrenorm : ', lamb1el)
    print('area under a2F unrenorm : ', trapz(a2F1, dx=dw1))
    print('lamb mass unrenorm : ', lamb1mass)
    print('lamb a2F renorm : ', lamb0)
    print('area under a2F renorm : ', trapz(a2F0, dx=dw0))
    #print('lamb electronic renorm : ', lamb0el)
    print('lamb mass renorm : ', lamb0mass)


    # McMillan Tc formula from a2F
    
    print('simple check of integral')
    print('lamb0', lamb0)
    dw = (w0[-1]-w0[0]) / (len(w0)-1)
    nr = len(a2F0)
    izero = np.argmin(np.abs(w0))
    lambcheck = 2 * np.sum(a2F0[izero+1:] / w0[izero+1:]) * dw
    print('lambcheck', lambcheck)
    lambcheckquad = 2 * np.sum(a2F0[izero+1:] / w0[izero+1:]) * dw
    L = len(w0[izero+1:])
    ws =  np.arange(0, dw*L, dw)
    print('lambcheck using quad : ', 2 * trapz(a2F0[izero+1:] / w0[izero+1:], dx=dw))
    
    print('normalization a2F renormalized : ', trapz(a2F0[izero+1:], dx=dw))
    print('normalization a2F unrenormalized : ', trapz(a2F1[izero+1:], dx=dw))
    
    
    wln = np.exp( 2 / lamb0 * np.sum(np.log(w0[izero+1:]) * a2F0[izero+1:] / w0[izero+1:]) * dw)
    print('wln ', wln)
    
    print('Tc ', wln/1.2 * np.exp(-1.04*(1 + lamb0) / lamb0))



compute_a2F()


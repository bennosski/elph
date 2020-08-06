
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
from a2F import corrected_a2F, corrected_lamb_mass, corrected_lamb_bare, corrected_a2F_imag, a2F_imag
import matplotlib.colors as mcolors
from scipy.integrate import trapz
from functions import read_params, find_folder

# a2F
def compute_a2F():
    #basedir = '/scratch/users/bln/elph/data/2dfixedn/'
    #basedir = '/scratch/users/bln/elph/data/single_iter/'
    #folder0 = 'data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    #folder1 = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'


    separate_imag_folder = False


    # original data for n=0.8 and fine freq spacing
    basedir = '/scratch/users/bln/elph/data/2d/'
    folder0 = 'data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    folder1 = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
    

 
    '''
    # new data for n=0.786 but idelta ~ 0.010 I think
    basedir = '/scratch/users/bln/elph/data/2dn0p786/'
    folder0 = 'data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.786_beta16.0000_QNone'
    folder1 = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.786_beta16.0000_QNone'
   
    
    # low temp data just beforce SC transition
    basedir = '/scratch/users/bln/elph/data/sc2dfixed/'
    folder0 = 'data_renormalized_nk200_abstp0.300_dim2_g00.33665_nw256_omega0.170_dens0.800_beta75.0000_QNone'
    folder1 = folder0 
    '''


    '''    
    # new larger frequency range data but idelta ~ 0.008 and 64x64
    basedir = '/scratch/users/bln/elph/data/debug/'
    params = {'renormalized': True, 'nk': 64, 'nw': 128, 'omega': 0.170, 'beta': 16.0, 'idelta': 0.050, 'wmin': -6.0, 'wmax': 10.0}
    _, folder0 = find_folder(basedir, params)
    folder1 = folder0
    separate_imag_folder = True
    '''
        
    params = read_params(basedir, folder0)
    print('renormalized:')
    print('dw', params['dw'])
    print('idelta', params['idelta'])

    params = read_params(basedir, folder1)
    print('unrenormalized')
    print('dw', params['dw'])
    print('idelta', params['idelta'])
    
    
    #for key in params:
    #    print(key, type(params[key]))

    print('folder', folder1)

    def main_computation():
        # main computation
        print('\n\n\nrenormalized lamb')
        _, lambbarekr, weightsr = corrected_lamb_bare(basedir, folder0, ntheta=80)
        lambmasskr, _ = corrected_lamb_mass(basedir, folder0, ntheta=80)
        lamba2Fkr, _ = corrected_a2F(basedir, folder0, ntheta=80)
        
        np.save(os.path.join(basedir, 'data/', folder0, 'lambbarekr'), lambbarekr)
        np.save(os.path.join(basedir, 'data/', folder0, 'lambmasskr'), lambmasskr)
        np.save(os.path.join(basedir, 'data/', folder0, 'lamba2Fkr'), lamba2Fkr)
        np.save(os.path.join(basedir, 'data/', folder0, 'weightsr'), weightsr)

        '''
        print('\n\n\nunrenormalized lamb')
        _, lambbareku, weightsu = corrected_lamb_bare(basedir, folder1, ntheta=80)
        lambmassku, _ = corrected_lamb_mass(basedir, folder1, ntheta=80)
        lamba2Fku, _ = corrected_a2F(basedir, folder1, ntheta=80)

        np.save(os.path.join(basedir, 'data/', folder1, 'lambbareku'), lambbareku)
        np.save(os.path.join(basedir, 'data/', folder1, 'lambmassku'), lambmassku)
        np.save(os.path.join(basedir, 'data/', folder1, 'lamba2Fku'), lamba2Fku)
        np.save(os.path.join(basedir, 'data/', folder1, 'weightsu'), weightsu)
        '''
    
    #############################
    # compute a2F
    #main_computation()    
    

    # compute lamb_a2F using imaginary axis only
    def imag():
        #lamba2Fikr, _ = corrected_a2F_imag(basedir, folder0, ntheta=80)
        #lamba2Fiku, _ = corrected_a2F_imag(basedir, folder1, ntheta=80)


        lamba2Fikr, _ = a2F_imag(basedir, folder0, ntheta=80, separate_imag_folder=separate_imag_folder)
        #lamba2Fiku, _ = a2F_imag(basedir, folder1, ntheta=80, separate_imag_folder=separate_imag_folder)

        
        '''
        if not separate_imag_folder:
            lamba2Fikr, _ = a2F_imag(basedir, folder0, ntheta=80)
            lamba2Fiku, _ = a2F_imag(basedir, folder1, ntheta=80)
        else:
            i1 = len(folder0)
            for _ in range(3):
                i1 = folder0.rfind('_', 0, i1-1)
            folder0_ = folder0[:i1]

            i1 = len(folder1)
            for _ in range(3):
                i1 = folder1.rfind('_', 0, i1-1)
            folder1_ = folder1[:i1]

            lamba2Fikr, _ = a2F_imag(basedir, folder0_, ntheta=80)
            #lamba2Fiku, _ = a2F_imag(basedir, folder1_, ntheta=80)
        '''


    print('')
    #imag()        
    print('')

    lamba2Fikr = np.load(basedir + 'data/' + folder0 + '/lambk_a2F_imag.npy')
    lamba2Fiku = np.load(basedir + 'data/' + folder1 + '/lambk_a2F_imag.npy')

    path0 = os.path.join(basedir, 'data/', folder0)
    path1 = os.path.join(basedir, 'data/', folder1)

    
    lambbarekr = np.load(os.path.join(path0, 'lambbarekr.npy'))
    lambmasskr = np.load(os.path.join(path0, 'lambmasskr.npy'))
    lamba2Fkr  = np.load(os.path.join(path0, 'lamba2Fkr.npy'))
    weightsr   = np.load(os.path.join(path0, 'weightsr.npy'))
    lambbareku = np.load(os.path.join(path1, 'lambbareku.npy'))
    lambmassku = np.load(os.path.join(path1, 'lambmassku.npy'))
    lamba2Fku  = np.load(os.path.join(path1, 'lamba2Fku.npy'))
    weightsu   = np.load(os.path.join(path1, 'weightsu.npy'))



    f = plt.figure()
    ntheta = len(lambmasskr)//4
    dtheta = np.pi / (2 * ntheta)
    thetas = np.arange(dtheta/2, np.pi/2, dtheta)
    print('ntheta', ntheta, 'len thetas', len(thetas))
    plt.plot(thetas * 180/np.pi, lambmasskr[:ntheta], 'k')
    
    ntheta = len(lamba2Fikr)
    dtheta = np.pi / (2 * ntheta)
    thetas = np.arange(dtheta/2, np.pi/2, dtheta)
    plt.plot(thetas * 180/np.pi, lamba2Fikr, 'k', linestyle='dotted')
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
    
    plt.savefig(basedir+'lambikr')
    plt.close()


    
    
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


    print('\nimag axis values:')
    a2Fu = np.load(basedir + 'data/' + folder1 + '/lamb_a2F_imag.npy')
    print('lamb a2F imag unrenorm : ', a2Fu)
    a2Fr = np.load(basedir + 'data/' + folder0 + '/lamb_a2F_imag.npy')
    print('lamb a2F imag renorm : ', a2Fr)     
    print('\n')

    
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
    legend(['$\lambda_{ME}$'+'={:.1f}'.format(a2Fu), '$\lambda_{RME}$'+'={:.1f}'.format(a2Fr)], fontsize=14, loc=2)
    xlim(0, 0.25)
    #ylim(0, 0.8)
    ylim(0, 2.0)
    xlabel('$\omega / t$', fontsize=13)
    ylabel(r'$\alpha^2 F$', fontsize=13)
    savefig(basedir+'a2Fcomp', transparent=True)
    close()

    dw0 = (w0[-1]-w0[0]) / (len(w0)-1)
    dw1 = (w1[-1]-w1[0]) / (len(w1)-1)

    print('')    
    print('lamb a2F unrenorm : ', lamb1)
    #print('lamb bare unrenorm : ', lamb1el)
    print('area under a2F unrenorm : ', trapz(a2F1, dx=dw1))
    print('lamb mass unrenorm : ', lamb1mass)

    print('')
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

    print('Tc from imag axis lambda', wln/1.2 * np.exp(-1.04*(1 + a2Fr) / a2Fr))




compute_a2F()


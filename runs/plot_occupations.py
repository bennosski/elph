import src
import os
from functions import read_params
import numpy as np
import matplotlib.pyplot as plt


def el_occ(basedir, folder):
    # electronic occupation as a test

    params = read_params(basedir, folder)
    
    #print(params['wmin'], params['wmax'])
    w = np.arange(params['wmin'], params['wmax'], params['dw'])
    
    G = np.load(os.path.join(basedir, 'data', folder, 'GR.npy'))
    #G = np.load(basedir + folder + '/GR.npy')
    
    nF = 1/(np.exp(params['beta']*w) + 1)
    norm = -1/np.pi * np.trapz(G.imag, dx=params['dw'], axis=2)
    nk = -1/np.pi * np.trapz(nF[None,None,:]*G.imag, dx=params['dw'], axis=2)

    plt.figure()
    plt.imshow(nk, origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig(basedir + 'nk')
    plt.close()
    
    plt.figure()
    plt.imshow(norm, origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig(basedir + 'norm')
    plt.close()
    

def ph_occ(basedir, folder):
    # electronic occupation as a test
    print(folder)

    params = read_params(basedir, folder)
    
    print(params['wmin'], params['wmax'], params['dw'])
    w = np.arange(params['wmin'], params['wmax'], params['dw'])
    dw = params['dw']

    D = np.load(os.path.join(basedir, 'data', folder, 'DR.npy'))

    if np.shape(D)[2]==840:
        print('wtf nr')
        w = np.arange(params['wmin'], params['wmax'], 0.010)
        dw = 0.010

    nB = 1/(np.exp(params['beta']*w) - 1)

    plt.figure()
    plt.plot(w, -1/np.pi * nB*D[0,0,:].imag)
    plt.savefig(basedir + 'DI' + folder[5])
    plt.close()    

    nr = np.shape(D)[2]
    I = -1/np.pi * np.trapz(D.imag[:,:,:nr//2], dx=dw, axis=2)
    print('integral of DI from -inf to 0 (avg, min, max)', np.mean(I), np.amin(I), np.amax(I))

    N = -1/np.pi * np.trapz(nB[None,None,:]*D.imag, dx=dw, axis=2)

    print('integral of nB DI (avg, min, max)', np.mean(N), np.amin(N), np.amax(N))

    N = (N - 1)/2

    print('N (avg, min, max)', np.mean(N), np.amin(N), np.amax(N))

    np.save(basedir + 'data/' + folder + '/Nphonon.npy', N)
    
    plt.figure()
    plt.imshow(N, origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig(basedir + 'data/' + folder + '/N')
    plt.close()


def plot_N_cut(basedir, folder0, folder1):

    params = read_params(basedir, folder0)    
  
    N0 = np.load(basedir + 'data/' + folder0 + '/Nphonon.npy')
    N1 = np.load(basedir + 'data/' + folder1 + '/Nphonon.npy')

    def get_N_cut(N):
        nk = params['nk']
        Ncut = []
        for i in range(nk//2):
            Ncut.append(N[i,i])
        for i in range(nk//2):
            Ncut.append(N[nk//2-i, nk//2])
        for i in range(nk//2):
            Ncut.append(N[0, nk//2-i])
        Ncut.append(N[0,0])
        return Ncut

    Ncut0 = get_N_cut(N0)
    Ncut1 = get_N_cut(N1)

    plt.figure()
    plt.plot(Ncut0)
    plt.plot(Ncut1)
    plt.savefig(basedir + 'Nphonon_cut')
    plt.close()



basedir = '/scratch/users/bln/elph/data/2d/'

folder0 = 'data_renormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
#el_occ(basedir, folder)
ph_occ(basedir, folder0)

folder1 = 'data_unrenormalized_nk120_abstp0.300_dim2_g00.33665_nw128_omega0.170_dens0.800_beta16.0000_QNone'
ph_occ(basedir, folder1)

plot_N_cut(basedir, folder0, folder1)

############################3

basedir = '/scratch/users/bln/elph/data/sc2dfixed/'

folder0 = 'data_renormalized_nk240_abstp0.300_dim2_g00.33665_nw256_omega0.170_dens0.800_beta100.0000_QNone'
#el_occ(basedir, folder)
ph_occ(basedir, folder0)

folder1 = 'data_unrenormalized_nk240_abstp0.300_dim2_g00.33665_nw256_omega0.170_dens0.800_beta100.0000_QNone'
ph_occ(basedir, folder1)

plot_N_cut(basedir, folder0, folder1)



import numpy as np
from scipy import interpolate


class Interp:

    def __init__(self, folder, Nk):
        # folder is the location where previous data is stored

        S = np.load(folder + '/S.npy')
        PI = np.load(folder + '/PI.npy')
        nk = np.load(folder + '/nk.npy')[0]
        ntau = np.shape(S)[2]
 
        dk = 2*np.pi/nk
        ks = np.arange(-np.pi, np.pi + dk/2, dk)
        Sext = np.concatenate((S, S[:,0,:,:,:][:,None,:,:,:]), axis=1)
        Sext = np.concatenate((Sext, Sext[0,:,:,:,:][None,:,:,:,:]), axis=0)

        PIext = np.concatenate((PI, PI[:,0,:][:,None,:]), axis=1)
        PIext = np.concatenate((PIext, PIext[0,:,:][None,:,:]), axis=0)

        ksint = np.arange(-np.pi, np.pi, 2*np.pi/Nk)

        self.S = np.zeros([Nk, Nk, ntau, 2, 2], dtype=complex)
        for it in range(ntau):
            for a in range(2):
                for b in range(2):
                    Ir = interpolate.interp2d(ks, ks, Sext[:,:,it,a,b].real, kind='linear')
                    Ii = interpolate.interp2d(ks, ks, Sext[:,:,it,a,b].imag, kind='linear')
                    self.S[:,:,it,a,b] = Ir(ksint, ksint) + 1j*Ii(ksint, ksint)

        self.PI = np.zeros([Nk, Nk, ntau], dtype=complex)
        for it in range(ntau):
            Ir = interpolate.interp2d(ks, ks, PIext[:,:,it].real, kind='linear')
            Ii = interpolate.interp2d(ks, ks, PIext[:,:,it].imag, kind='linear')
            self.PI[:,:,it] = Ir(ksint, ksint) + 1j*Ii(ksint, ksint)
      


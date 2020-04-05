import numpy as np
from scipy import interpolate


class Interp:

    def __init__(self, folder, Nk):
        # folder is the location where previous data is stored

        S = np.load(folder + '/S.npy')
        PI = np.load(folder + '/PI.npy')
        nk = np.load(folder + '/nk.npy')[0]
        ntau = np.shape(S)[2]
 
        ks = np.arange(-np.pi, np.pi, 2*np.pi/nk)

        ksint = np.arange(-np.pi, np.pi, 2*np.pi/Nk)

        self.S = np.zeros([Nk, Nk, ntau, 2, 2], dtype=complex)
        for it in range(ntau):
            for a in range(2):
                for b in range(2):
                    I = interpolate.interp2d(ks, ks, S, kind='linear')
                    self.S[:,:,it,a,b] = I(ksint, ksint)

        self.PI = np.zeros([Nk, Nk, ntau], dtype=complex)
        for it in range(ntau):
            for a in range(2):
                for b in range(2):
                    I = interpolate.interp2d(ks, ks, PI, kind='linear')
                    self.PI[:,:,it,a,b] = I(ksint, ksint)

      


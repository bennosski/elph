import numpy as np
from scipy import interpolate


class Interp:

    def __init__(self, folder, arg, kind=None):
        if kind is None or kind=='momentum':
            self._interp_momentum(folder, arg)
        elif kind=='frequency':
            self._interp_frequency(folder, arg) 


    def _interp_frequency(self, folder, W):
        nk = np.load(folder + '/nk.npy')[0]
        w  = np.load(folder + '/w.npy')

        SR = np.load(folder + '/SR.npy')
        PIR = np.load(folder + '/PIR.npy')

        Ir = interpolate.interp1d(w, SR.real, axis=-1, kind='linear')
        Ii = interpolate.interp1d(w, SR.imag, axis=-1, kind='linear')
        self.SR = Ir(W) + 1j*Ii(W)

        Ir = interpolate.interp1d(w, PIR.real, axis=-1, kind='linear')
        Ii = interpolate.interp1d(w, PIR.imag, axis=-1, kind='linear')
        self.PIR = Ir(W) + 1j*Ii(W)


    def _interp_momentum(self, folder, Nk):
        # folder is the location where previous data is stored

        print('loading from ', folder)

        dim = int(np.load(folder + '/dim.npy'))
        sc = bool(np.load(folder + '/sc.npy'))
        S = np.load(folder + '/S.npy')
        PI = np.load(folder + '/PI.npy')
        nk = np.load(folder + '/nk.npy')[0]
        ntau = np.shape(S)[dim]
 
        dk = 2*np.pi/nk
        ks = np.arange(-np.pi, np.pi + dk/2, dk)
        
        if dim==2:
            Sext = np.concatenate((S, S[:,0,...][:,None,...]), axis=1)
            Sext = np.concatenate((Sext, Sext[0,...][None,...]), axis=0)
    
            PIext = np.concatenate((PI, PI[:,0,:][:,None,:]), axis=1)
            PIext = np.concatenate((PIext, PIext[0,...][None,...]), axis=0)
        else:
            raise NotImplementedError


        ksint = np.arange(-np.pi, np.pi, 2*np.pi/Nk)

        if sc:
            self.S = np.zeros([Nk, Nk, ntau, 2, 2], dtype=complex)
            for it in range(ntau):
                for a in range(2):
                    for b in range(2):
                        Ir = interpolate.interp2d(ks, ks, Sext[:,:,it,a,b].real, kind='linear')
                        Ii = interpolate.interp2d(ks, ks, Sext[:,:,it,a,b].imag, kind='linear')
                        self.S[:,:,it,a,b] = Ir(ksint, ksint) + 1j*Ii(ksint, ksint)
    
        else:
            self.S = np.zeros([Nk, Nk, ntau], dtype=complex)
            for it in range(ntau):
                Ir = interpolate.interp2d(ks, ks, Sext[:,:,it].real, kind='linear')
                Ii = interpolate.interp2d(ks, ks, Sext[:,:,it].imag, kind='linear')
                self.S[:,:,it] = Ir(ksint, ksint) + 1j*Ii(ksint, ksint)

    
        self.PI = np.zeros([Nk, Nk, ntau], dtype=complex)
        for it in range(ntau):
            Ir = interpolate.interp2d(ks, ks, PIext[:,:,it].real, kind='linear')
            Ii = interpolate.interp2d(ks, ks, PIext[:,:,it].imag, kind='linear')
            self.PI[:,:,it] = Ir(ksint, ksint) + 1j*Ii(ksint, ksint)
          



import src
import numpy as np
import fourier
import time

beta = 10.0
nk = 64
nw = 1024
x = np.random.randn(nk, nk, nw)

time0 = time.time()
y = np.apply_along_axis(fourier.t2w_fermion_alpha0, 2, x, beta)
print('took ', time.time()-time0)


time0 = time.time()
x_ = np.concatenate((x, x), axis=2)
z = np.fft.fft(x_, axis=2)
print('took ', time.time()-time0)




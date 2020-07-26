# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:32:06 2020

@author: benno
"""


# lambm / lamb0 as a function of temperature


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import polygamma
from mpmath import psi


def get_x_y(omega):
    
    Ts = []
    ys = []
    
    for T in np.linspace(0.01, 2, 200):    
        x = omega * 1j / (2*np.pi*T)    
        Ts.append(T)
        ys.append(-omega / (4*np.pi*T) * (psi(1, 0.5 + x) - psi(1, 0.5 - x)).imag)

    return np.array(Ts), np.array(ys)

plt.figure()
Ts, ys = get_x_y(1)
plt.plot(Ts, ys)
plt.xlim(0, 2)
plt.ylim(0, 1.3)
#plt.ylabel('y')
plt.ylabel(r'${\lambda_m}/\lambda_{\alpha^2F}$', fontsize=15)
plt.xlabel(r'T / $\Omega$', fontsize=14)
plt.savefig('lambmvslamba2F')
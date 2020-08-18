from numpy import *
import matplotlib
from scipy import optimize
from matplotlib.pyplot import *
from scipy.interpolate import UnivariateSpline
from scipy import integrate
import numpy as np


def E(kx, ky, kz):
    return -2.0*(cos(kx) + cos(ky) + cos(kz)) 

def Ecut(r, theta, phi, mu):
    return E(r*sin(theta)*cos(phi), r*sin(theta)*sin(phi),  r*cos(theta)) - mu


def compute_fill_naive(mu, nk):
    ks = np.arange(0, np.pi, np.pi/nk)
    band = E(ks[:,None,None], ks[None,:,None], ks[None,None,:]) - mu    
    #return 2 * np.sum(1/(np.exp(20*band) + 1)) / nk**3
    return 2 * np.sum(band < 0) / nk**3
    

def compute_fill(mu):

    def dAdtheta(theta, phi):
        r = optimize.brentq(Ecut, 0, pi, args=(theta, phi, mu))
        return r**3/3.0 * sin(theta)

    area, err = integrate.dblquad(dAdtheta, 0, np.pi/4, lambda theta : 0, lambda theta : np.pi/2)
    area *= 2
    
    n = area/pi**3 * 2.0
    return n


def compute_spline(mus):
    # compute spline around mu
    
    fills = array([compute_fill(mu) for mu in mus])

    deriv = 0.5 * (fills[1:] - fills[:-1]) / (mus[1:] - mus[:-1])
    mus_centered = (mus[1:] + mus[:-1]) / 2.0
    fills_centered = (fills[1:] + fills[:-1]) / 2.0
    
    figure()
    plot(mus_centered, deriv)
    uspl = UnivariateSpline(mus_centered, deriv, s=0)
    plot(mus_centered, uspl(mus_centered))
    xlabel('mu')
    ylabel('dos')
    ylim(0, gca().get_ylim()[1])
    show()
    
    figure()
    plot(fills_centered, deriv)
    fspl = UnivariateSpline(fills_centered, deriv, s=0)
    plot(fills_centered, fspl(fills_centered))
    xlabel('fill')
    ylabel('dos')
    ylim(0, gca().get_ylim()[1])
    show()    
    
    return uspl, fspl
    


def dos_from_delta_fcns(tp, fill, nk, gamma):
    mu = optimize.minimize(lambda mu : np.abs(compute_fill2(mu, tp) - fill), -1.11).x[0]
    print('mu at n=0.8 : ', mu)
    
    ks = np.arange(-np.pi, np.pi, 2*np.pi/nk)
    kx, ky = np.meshgrid(ks, ks)
    band = E(kx, ky, tp) - mu  
    
    figure()
    imshow(band < 0)

    deltas = gamma/np.pi / (band**2 + gamma**2)
    
    return np.sum(deltas) / nk**2


# tests showing the compute_fill via area integration works really well
# naive method extrapolates to the predicted value
'''
nks = np.arange(10, 80, 10)
fill_vs_nk = [compute_fill_naive(-2, nk) for nk in nks]

figure()
plot(nks, fill_vs_nk, '.-')
x1, x2 = gca().get_xlim()
fill = compute_fill(-2)
hlines(fill, 0, x2)

figure()
plot(1/nks, np.array(fill_vs_nk) - fill)
x1,x2 = gca().get_xlim()
xlim(0, x2)
y1,y2 = gca().get_ylim()
ylim(0, y2)
'''

'''
print('filling = ', compute_fill(-2))
uspl, fspl = compute_spline(np.linspace(-2.1, -2.0, 10))
print('dos = ', uspl(-2))
'''

#fill = 0.4
#mu = optimize.minimize(lambda mu : np.abs(compute_fill(mu) - fill), -2, tol=1e-3).x[0]
#print('mu = ', mu)
mu = -2.103700128624296
print('corresponding filling ', compute_fill(mu))
uspl, fspl = compute_spline(np.linspace(mu-0.1, -2.0, 20))
print('dos ', uspl(mu))


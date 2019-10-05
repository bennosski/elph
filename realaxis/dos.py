from numpy import *
from scipy import optimize
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.pyplot import *
from scipy.interpolate import UnivariateSpline
from scipy import integrate

def E(kx, ky, tp=0.0):
    return -2.0*(cos(kx) + cos(ky)) - 4.0*tp*cos(kx)*cos(ky)

def Ecut(r, theta, mu, x0, y0, sgn, tp):
    return E(x0+sgn*r*cos(theta), y0+sgn*r*sin(theta), tp) - mu

def compute_fill(mu, tp):

    E00   = E(0,0,tp)   - mu
    Epipi = E(pi,pi,tp) - mu
    Epi0  = E(pi,0,tp)  - mu
    
    if Epi0*Epipi <= 0.0:
        x0,y0,sgn = pi,pi,-1
    elif E00*Epi0 < 0.0:
        x0,y0,sgn = 0,0,1
    else:
        print('error signs')
        print(E00, Epi0, Epipi)
        exit()

    def dAdtheta(theta, mu, x0, y0, sgn, tp):
        r = optimize.brentq(Ecut, 0, pi/cos(theta), args=(theta, mu, x0, y0, sgn, tp))
        return r**2/2.0

    area, err = integrate.quad(dAdtheta, 0, pi/4, args=(mu, x0, y0, sgn, tp))
    
    if sgn==-1:
        n = (1.0 - 2.0*area/pi**2) * 2.0
    else:
        n = 2.0*area/pi**2 * 2.0
    return n

def compute_n_vs_mu(tp):
    def Eband(x, tp, minimize):
        return minimize * E(x[0], x[1], tp)
    
    kx, ky = optimize.minimize(Eband, array([0,0]), args=(tp, 1)).x
    #print(kx, ky)
    #print('mu_min', E(kx,ky,tp))
    mu_min = E(kx,ky,tp)

    kx, ky = optimize.minimize(Eband, array([0,0]), args=(tp, -1)).x
    #print(kx, ky)
    #print('mu_max', E(kx,ky,tp))
    mu_max = -E(kx,ky,tp)
    
    mus = linspace(mu_min+0.01, mu_max-0.01, 200)    
    fills = array([compute_fill(mu, tp) for mu in mus])

    return mus, fills

def compute_spline_mu_vs_n(tp):
    mus, fills = compute_n_vs_mu(tp)

    figure()
    plot(mus, fills)
    xlabel('mu')
    ylabel('fill')
    show()

    return UnivariateSpline(fills, mus, s=0)

def compute_spline_dos(tp):
    mus, fills = compute_n_vs_mu(tp) 

    deriv = 0.5 * (fills[1:] - fills[:-1]) / (mus[1:] - mus[:-1])
    mus_centered = (mus[1:] + mus[:-1]) / 2.0

    figure()
    plot(mus_centered, deriv)
    spl = UnivariateSpline(mus_centered, deriv, s=0)
    plot(mus_centered, spl(mus_centered))
    xlabel('mu')
    ylabel('dos')
    ylim(0, gca().get_ylim()[1])
    show()

    return spl

if __name__=='__main__':
    #tp = 0.0
    tp = -0.3

    spl = compute_spline_mu_vs_n(tp)
    #fill = 0.4
    fill = 0.8
    print('mu at fill={:.3f} = '.format(fill), spl(fill))
    print('EF = E(0,0)-mu = {:.16f}'.format(E(0,0,tp)-spl(fill)))
    
    #spl = compute_spline_dos(tp)



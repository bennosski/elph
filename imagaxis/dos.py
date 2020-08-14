from numpy import *
import matplotlib
#matplotlib.use('TkAgg')
#from params import params
from scipy import optimize
from matplotlib.pyplot import *
from scipy.interpolate import UnivariateSpline
from scipy import integrate

def E(kx, ky, tp=0.0):
    return -2.0*(cos(kx) + cos(ky)) - 4.0*tp*cos(kx)*cos(ky)

def Ecut(r, theta, mu, x0, y0, sgn, tp):
    return E(x0+sgn*r*cos(theta), y0+sgn*r*sin(theta), tp) - mu

def compute_fill(mu, ntheta, tp):

    E00   = E(0,0,tp)   - mu
    Epipi = E(pi,pi,tp) - mu
    Epi0  = E(pi,0,tp)  - mu
    
    if Epi0*Epipi <= 0.0:
        x0,y0,sgn = pi,pi,-1
    elif E00*Epi0 < 0.0:
        x0,y0,sgn = 0,0,1
    else:
        print('error signs')
        exit()
    
    thetas = linspace(0, pi/4, ntheta)
    rs = []
    for theta in thetas:
        r = optimize.brentq(Ecut, 0, pi/cos(theta), args=(theta, mu, x0, y0, sgn, tp))
        rs.append(r)

    rs = array(rs)
                
    weights = ones(len(rs))
    weights[0] = 0.5
    weights[-1] = 0.5
    
    dtheta = pi/4 / (ntheta-1)
    area = 0.5 * sum(weights * array(rs)**2) * dtheta
    if sgn==-1:
        n = (1.0 - 2.0*area/pi**2) * 2.0
    else:
        n = 2.0*area/pi**2 * 2.0
    return n


def compute_spline(tp):
    ntheta = 300
    
    mus = linspace(-3.99, 0, 100)

    fills = [compute_fill(mu, ntheta, tp) for mu in mus]

    figure()
    plot(mus, fills)
    title('n vs mu')

    figure()
    plot(fills, mus)
    title('mu vs n')

    dmu = (mus[-1]-mus[0])/(len(mus)-1)
    fills = array(fills)
    dos = 0.5 * (fills[1:]-fills[:-1])/(dmu)
    mus_centered = (mus[1:] + mus[:-1])/2.0

    figure()
    plot(mus_centered, dos)
    xlabel('mu')
    ylabel('dos')
    ylim(0, gca().get_ylim()[1])
    show()



def compute_fill2(mu, tp):

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


def compute_spline2(tp):
    
    def Eband(x, tp, sgn):
        return sgn * E(x[0], x[1], tp)
    
    kx, ky = optimize.minimize(Eband, array([0,0]), args=(tp, 1)).x
    print(kx, ky)
    mu_min = E(kx, ky, tp)
    print('mu_min', mu_min)

    kx, ky = optimize.minimize(Eband, array([np.pi,np.pi]), args=(tp, -1)).x
    print(kx, ky)
    mu_max = E(kx, ky, tp)
    print('mu_max', mu_max)
    
    mus = linspace(mu_min*0.9, mu_max*0.9, 200)    
    fills = array([compute_fill2(mu, tp) for mu in mus])

    print('mu at n=0.8 : ', optimize.minimize(lambda mu : np.abs(compute_fill2(mu, tp) - 0.8), -1.11).x[0])


    deriv = 0.5 * (fills[1:] - fills[:-1]) / (mus[1:] - mus[:-1])
    mus_centered = (mus[1:] + mus[:-1]) / 2.0
    fills_centered = (fills[1:] + fills[:-1]) / 2.0
    
    
    figure()
    plot(mus_centered, deriv)
    spl = UnivariateSpline(mus_centered, deriv, s=0)
    plot(mus_centered, spl(mus_centered))
    xlabel('mu')
    ylabel('dos')
    ylim(0, gca().get_ylim()[1])
    show()

    
    figure()
    plot(fills_centered, deriv)
    spl = UnivariateSpline(fills_centered, deriv, s=0)
    plot(fills_centered, spl(fills_centered))
    xlabel('fill')
    ylabel('dos')
    ylim(0, gca().get_ylim()[1])
    show()    
    
    print('dos at n=0.4 : ', spl(0.4))
    print('dos at n=0.8 : ', spl(0.8))


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



tp = 0.0
fill = 0.4
print('dos deltas : ', dos_from_delta_fcns(tp, fill, 800, 0.01))
compute_fill2(-1.11, tp)
compute_spline2(tp)


'''
tp = -0.3
fill = 0.8
print('dos deltas : ', dos_from_delta_fcns(tp, fill, 800, 0.01))

compute_fill2(-1.11, tp)
compute_spline2(-0.3)
'''

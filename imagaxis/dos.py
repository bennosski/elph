from numpy import *
from params import params
from scipy import optimize

band = params['band']
Nk = 400

ek = band(Nk)

# initial guess for mu...

#def fill(mu):
#    return 2*sum(ek-mu<=0.0, axis=None) / Nk**2

#mu0 = optimize.fsolve(lambda mu: fill(mu)-params['dens'], -1.2)



def fill(mu):
    return 2.0*mean(1.0/(exp(90.0*(ek-mu))+1.0))

mu0 = optimize.fsolve(lambda mu: fill(mu)-params['dens'], -1.12)

print(mu0)
print(fill(mu0))
    
dmu = 0.01

f1 = fill(mu0+dmu)
f2 = fill(mu0)
f3 = fill(mu0-dmu)

print('diff', f1-f3)
print('f1, f2, f3', f1, f2, f3)

print('DOS estimate ', -0.5*(f1 - 2*f2 + f3)/dmu**2)

from matplotlib.pyplot import *
x = ek-mu
imshow(x<0.0)
show()

#figure()
#hist

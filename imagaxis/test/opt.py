import src
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.pyplot import *
from scipy import optimize
import numpy as np
from anderson import AndersonMixing


fun = lambda x : x[0] + np.sin(x[1])

cons = ({'type': 'eq', 'fun': lambda x : x[0]**2 + x[1]**2 - 0.5**2})
x = optimize.minimize(fun, (0,0), method='SLSQP', constraints=cons).x
print(x)
print(np.sqrt(np.sum(x**2)))

#------------------------------------

AM = AndersonMixing(0.01)

#g = lambda x : np.cos(x)-0.5
#g = lambda x : np.cos(x)+0.5
#g = lambda x : 0.5*x + 3.0

# example of overshoot in naive
#g = lambda x : -2.0*x**2 + 2.0
g = lambda x : -2.0*x + 2.0

x = 0.1
errs = []

errs_naive = []
xnaive = 0.0

errs_linear = []
xlinear = 0.0

for _ in range(15):
    try:
        xnew = g(xnaive)   
        errs_naive.append(np.abs(xnew-xnaive))
        xnaive = xnew
        print('xnaive', xnaive)
    
        xold = xlinear
        xnew = g(xlinear)
        errs_linear.append(np.abs(xnew-xlinear))
        xlinear = 0.9*xnew + 0.1*xold
        print('xlinear', xlinear)
    except:
        pass

    
    x = AM.step(x, g(x))
    print('x', x)
    print('error ', g(x) - x)
    print('')
    errs.append(np.abs(g(x) - x))

figure()
semilogy(errs_naive, '.-')
semilogy(errs_linear, '.-')
semilogy(errs, '.-')
ylim(gca().get_ylim()[0], np.log(np.amax(errs_linear))+100.0)
legend(['naive', 'linear', 'anderson'])
ylabel('err')
xlabel('iteration')
show()

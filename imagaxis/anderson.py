from scipy.optimize import minimize
import numpy as np


class AndersonMixing:

    def __init__(self, m):
        assert m >= 1
        self.m = m
        self.gs = {}
        self.fs = {}
        self.L = 0

        self.cons = ({'type': 'eq', 'fun': lambda alphas : np.sum(alphas) - 1.0})

    def step(self, x, gx):
        '''
        fixed point iteration for a function g(x)
        x is the current point
        gx is g(x)
        ''' 

        L = self.L
        m = self.m

        if L>=m:
           del self.gs[L-m]
           del self.fs[L-m]

        x, gx = np.array(x), np.array(gx)

        self.gs[L] = gx        
        self.fs[L] = np.reshape(gx - x, [-1,1])
        
        mk = min(m, L+1)
        fs = np.concatenate([self.fs[L-mk+1+i] for i in range(mk)], axis=1)

        alphas0 = np.ones(mk) / np.sqrt(mk)
        alphas  = minimize(lambda alphas : np.linalg.norm(fs @ alphas), alphas0, method='SLSQP', constraints=self.cons).x  

        #print('fs', fs)
        print('alphas ', alphas, ' norm ', np.sum(alphas))

        self.L += 1
        return np.sum([alphas[i]*self.gs[L-mk+1+i] for i in range(mk)], axis=0)
   
    
    def step2(self, x, gx):
        '''
        fixed point iteration for a function g(x)
        x is the current point
        gx is g(x)
        ''' 

        L = self.L
        m = self.m

        if L>=m:
           del self.gs[L-m]
           del self.fs[L-m]

        self.gs[L] = np.array(gx)        
        self.fs[L] = np.linalg.norm(gx - x)**2
        
        mk = min(m, L+1)
        fs = np.array([self.fs[L-mk+1+i] for i in range(mk)])

        alphas0 = np.ones(mk) / np.sqrt(mk)
        alphas  = minimize(lambda alphas : np.sum(fs @ alphas**2), alphas0, method='SLSQP', constraints=self.cons).x  

        #print('mk ', mk)
        #print('fs', fs)
        print('alphas ', alphas, ' norm ', np.sum(alphas))

        self.L += 1
        return np.sum([alphas[i]*self.gs[L-mk+1+i] for i in range(mk)], axis=0)
   
    '''
    def step(self, x, gx):
        L = self.L
        m = self.m

        if L>=m:
           del self.gs[L-m]
           del self.fs[L-m]

        x, gx = np.array(x), np.array(gx)

        self.gs[L] = gx.copy()
      
        x, gx = np.reshape(x, [-1,1]), np.reshape(gx, [-1,1])
        self.fs[L] = np.reshape(np.linalg.norm(gx - x)**2, [-1,1]) #sum(gxmx * np.conj(gxmx))

        i0 = max(0, L-m+1)
        fs = np.concatenate([self.fs[i] for i in range(i0, L+1)], axis=1)    
        mk = np.shape(fs)[1]
        alphas0 = np.ones(mk) / np.sqrt(mk)
        alphas = minimize(lambda alphas : np.sum(fs @ alphas**2), alphas0, method='SLSQP', constraints=self.cons).x  

        print('mk ', mk)
        print('fs', fs)
        print('alphas ', alphas, ' norm ', np.sum(alphas))

        self.L += 1

        return np.sum([alphas[i]*self.gs[i0+i] for i in range(mk)])
'''   

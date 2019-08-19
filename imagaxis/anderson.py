from scipy.optimize import minimize
import numpy as np

class AndersonMixing:

    def __init__(self, alpha, frac=0.1):
        '''
        frac is linear mixing fraction for first two iterations
        alpha for anderson mixing
        '''
        self.alpha = alpha
        self.frac  = frac
        self.itr = 0
        
    def step(self, p, F):
        # convergence check
        if np.sum(np.abs(p-F))<1e-15:
            return F

        # linear mixing for first two iterations
        if self.itr == 0 or self.itr==1:
            self.p = p
            self.F = F
            self.H = F-p
            self.itr += 1
            return (1-self.frac)*F + self.frac*p

        # Anderson mixing
        self.pp = self.p
        self.Fp = self.F
        self.Hp = self.H
        self.p = p
        self.F = F
        self.H = F - p
        dH = self.H - self.Hp
        theta = - np.sum(dH * self.Hp) / np.sum(dH * dH)
        
        return self.pp + self.alpha*self.Hp + theta * (self.p - self.pp + self.alpha*dH)
        #return self.p + self.alpha*self.H + theta * (self.p - self.pp + self.alpha*dH
        

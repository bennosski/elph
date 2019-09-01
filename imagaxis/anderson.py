from scipy.optimize import minimize
import numpy as np

class AndersonMixing:

    def __init__(self, alpha, frac=0.1, n=2):
        '''
        frac is linear mixing fraction for first two iterations
        alpha for anderson mixing
        '''
        self.alpha = alpha
        self.frac  = frac
        self.n = n

        self.ps = []
        self.Fs = []
        self.Hs = []

    def step(self, p, F):
        '''
        p is input  -- x
        F is output -- f(x)
        '''

        # convergence check
        if np.sum(np.abs(p-F))<1e-15:
            return F

        self.ps.append(p)
        self.Fs.append(F)
        self.Hs.append(F-p)

        # linear mixing for first two iterations
        if len(self.ps)==1 or len(self.ps)==2:
            return (1-self.frac)*F + self.frac*p

        # Anderson mixing 
        if len(self.ps)>self.n:
            del self.ps[0]
            del self.Fs[0]
            del self.Hs[0]

        n = len(self.ps)
        R2 = np.zeros((n,n), dtype=complex)
        for i in range(n):
            for j in range(n):
                R2[i,j] = np.sum(np.conj(self.Hs[i]) * self.Hs[j])
        w,v = np.linalg.eigh(R2)
        v = v[:,0]
        v /= np.sum(v)
        #print('v', v)

        p_, F_ = 0, 0
        for i in range(n):
            p_ += np.conj(v[i]) * self.ps[i]
            F_ += np.conj(v[i]) * self.Fs[i]

        return (1-self.alpha)*p_ + self.alpha*F_


    def newstep(self, p, F):
        '''
        p is input  -- x
        F is output -- f(x)
        '''
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
        
        print('theta')
        print(theta)

        n = 2
        X = np.vstack((np.array([self.Hp]), np.array([self.H])))
        R2 = np.zeros((n,n))
        for a in range(n):
            for b in range(n):
                R2[a,b] = np.sum(X[a] * X[b])      
        w,v = np.linalg.eigh(R2)
        v = v[:,0]
        v /= np.sum(v)
        print('v', v)

        X = np.vstack((np.array([self.pp]), np.array([self.p])))
        p_ = np.einsum('a,a...->...', v, X)
        X = np.vstack((np.array([self.Fp]), np.array([self.F])))
        F_ = np.einsum('a,a...->...', v, X)

        out = (1-self.alpha)*p_ + self.alpha*F_

        out2 = self.pp + self.alpha*self.Hp + theta * (self.p - self.pp + self.alpha*dH)

        print('out')
        print(out)
        print('out2')
        print(out2)

        return self.pp + self.alpha*self.Hp + theta * (self.p - self.pp + self.alpha*dH)
        #return self.p + self.alpha*self.H + theta * (self.p - self.pp + self.alpha*dH) # wrong
        

        
    def old_step(self, p, F):
        '''
        p is input  -- x
        F is output -- f(x)
        '''
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
        #return self.p + self.alpha*self.H + theta * (self.p - self.pp + self.alpha*dH) # wrong
        

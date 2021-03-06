# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:29:12 2020

@author: 18595
"""


import src
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar
from scipy.interpolate import interp2d
from scipy.misc import derivative
import fourier
from functions import read_params
from migdal_2d import Migdal as RME2d
from migdal_2d_sc import Migdal as RME2dsc


# function to compute kF(theta)
def kF(theta, theta0, x0, y0, t, tp, mu):
    def r2k(r):
        return x0 + r*np.cos(theta0+theta), y0 + r*np.sin(theta0+theta)
        
    def ek(r):
        kx, ky = r2k(r)
        return -2.0*t*(np.cos(kx)+np.cos(ky)) - 4.0*tp*np.cos(kx)*np.cos(ky) - mu

    r = root_scalar(ek, bracket=(0, np.pi)).root
    #print('found kx, ky = ', r, r2k(r))
    deriv = derivative(ek, r, dx=1e-4)
    assert deriv < 0
    return r2k(r), r, deriv
    

def corrected_kF(Is, theta, theta0, x0, y0, t, tp, mu):
    I = Is[1] # this is SR(w=0, k)
    
    def r2k(r):
        return x0 + r*np.cos(theta0+theta), y0 + r*np.sin(theta0+theta)
        
    def ekpSR(r):
        kx, ky = r2k(r)
        ek = -2.0*t*(np.cos(kx)+np.cos(ky)) - 4.0*tp*np.cos(kx)*np.cos(ky) - mu
        return ek + I(kx, ky)[0]

    r = root_scalar(ekpSR, bracket=(0, np.pi)).root
    #print('found kx, ky = ', r, r2k(r))
    deriv = derivative(ekpSR, r, dx=1e-4)
    assert deriv < 0
    return r2k(r), r, deriv


# function to interpolate sigma for 3 freq points
def interpS(SR, wr, nr, nk, izero):
    print('shape SR', np.shape(SR))

    #ks = np.arange(-np.pi, np.pi, 2*np.pi/nk)
    ks = np.linspace(-np.pi, np.pi, nk+1)
    Is = []
    for iw in (-2, 0, 2):
        if len(np.shape(SR))==5:
            Ir = interp2d(ks, ks, SR[:,:,izero+iw,0,0].real, kind='linear')
        else:
            Ir = interp2d(ks, ks, SR[:,:,izero+iw].real, kind='linear')
        Is.append(Ir)
    return Is


# function to compute fermi velocity (need deriv sigma real)
def vel(kx, ky, dEdk, Is, wr, nr, izero):
    ys = [I(kx, ky)[0] for I in Is]
    dw = wr[izero+2] - wr[izero-2]
    # ys[2] is SR at iw=nr//2+2
    # ys[1] is SR at iw=nr//2
    # ys[0] is SR at iw=nr//2-2
    dSdE = (ys[2] - ys[0]) / dw
    #print('dSdE', dSdE)
    #print('dEdk', dEdk)
    assert dSdE < 0
    return np.abs(dEdk) / (1 - dSdE)


def vel_and_dSdE(kx, ky, dEdk, Is, wr, nr, izero):
    ys = [I(kx, ky)[0] for I in Is]
    dw = wr[izero+2] - wr[izero-2]
    # ys[2] is SR at iw=nr//2+2
    # ys[1] is SR at iw=nr//2
    # ys[0] is SR at iw=nr//2-2
    dSdE = (ys[2] - ys[0]) / dw
    #print('dSdE', dSdE)
    #print('dEdk', dEdk)
    assert dSdE < 0
    return np.abs(dEdk) / (1 - dSdE), dSdE

 


# final function for avg lambda with fixed omega
# final function for avg lambda with a^2F (I think this is the better way)
# use a bare D here and see how close these two definitions are
# will need small iomega for them to agree though


'''
(kx, ky), r, dEdk = kF(0, -np.pi, -np.pi, t, tp, mu)  
Is = interpS(SR, wr, nr, nk)
v = vel(kx, ky, dEdk, Is, wr, nr)
print(v)
'''


# procedure
# get all the k points of interest
# get all the velocities of interest
# compute B from DR
# get all the Bs of interest (nr/2 x nkf)
# for each k, compute FS integral over k'
# compute FS avg to get a2F

def lamb_bare(basedir, folder, ntheta=5):
    # compute lambda without a2F (assuming fixed omega)
    # this was wrong.....need to do second integral.....
    # this was lambda_k not full averaged lambda
    # only for the unrenormalized case
    
    wr, nr, nk, SR, DR, mu, t, tp, g0, omega = load(basedir, folder)
    
    izero = np.argmin(np.abs(wr))

    dtheta = np.pi/(2*ntheta)
    thetas = np.arange(dtheta/2, np.pi/2, dtheta)
    assert len(thetas) == ntheta
    
    corners = ((0,        -np.pi, -np.pi), 
               (-np.pi/2, -np.pi,  np.pi),
               (-np.pi,    np.pi,  np.pi),
               (np.pi/2,   np.pi, -np.pi))
    
    kxfs  = []
    kyfs  = []
    dEdks = []
    vels  = []
    rs    = []
    Is = interpS(SR, wr, nr, nk, izero)
    for corner in corners:
        for theta in thetas:
            (kx, ky), r, dEdk =  kF(theta, corner[0], corner[1], corner[2], t, tp, mu)
            kxfs.append(kx)
            kyfs.append(ky)
            rs.append(r)
            dEdks.append(dEdk)
            v = vel(kx, ky, dEdk, Is, wr, nr, izero)
            vels.append(v)
    
    lamb = 0
    for ik in range(ntheta):
        lamb += rs[ik] / vels[ik] * dtheta
    lamb *= 4
    
    lamb *= 2 * g0**2 / omega / (2*np.pi)**2

    print('lamb electronic = ', lamb)
    np.save(basedir + 'data/'+folder+'/lamb_electronic.npy', [lamb])
    
    return lamb, 2 * g0**2 / omega * np.ones(ntheta)

def corrected_lamb_bare(basedir, folder, ntheta=5):
    # compute lambda without a2F (assuming fixed omega)
    # only for the unrenormalized case
    # corrected for Fermi surface
    
    wr, nr, nk, SR, DR, mu, t, tp, g0, omega = load(basedir, folder)

    izero = np.argmin(np.abs(wr))
    
    dtheta = np.pi/(2*ntheta)
    thetas = np.arange(dtheta/2, np.pi/2, dtheta)
    assert len(thetas) == ntheta
    
    corners = ((0,        -np.pi, -np.pi), 
               (-np.pi/2, -np.pi,  np.pi),
               (-np.pi,    np.pi,  np.pi),
               (np.pi/2,   np.pi, -np.pi))
    
    kxfs  = []
    kyfs  = []
    dEdks = []
    vels  = []
    rs    = []
    Is = interpS(SR, wr, nr, nk, izero)
    for corner in corners:
        for theta in thetas:
            (kx, ky), r, dEdk =  kF(theta, corner[0], corner[1], corner[2], t, tp, mu)
            kxfs.append(kx)
            kyfs.append(ky)
            rs.append(r)
            dEdks.append(dEdk)
            #v = vel(kx, ky, dEdk, Is, wr, nr)
            vels.append(np.abs(dEdk))
    
    lamb = 0
    for ik in range(ntheta):
        lamb += rs[ik] / vels[ik] * dtheta
    lamb *= 4
    
    lamb *= 2 * g0**2 / omega / (2*np.pi)**2

    print('lamb bare = ', lamb)
    np.save(basedir + 'data/'+folder+'/lamb_bare.npy', [lamb])
    
    return lamb, lamb * np.ones(4*ntheta), np.array(rs)/np.array(vels)


def lamb_mass(basedir, folder, ntheta=5):
    
    wr, nr, nk, SR, DR, mu, t, tp, g0, omega = load(basedir, folder)

    izero = np.argmin(np.abs(wr))
    
    dtheta = np.pi/(2*ntheta)
    thetas = np.arange(dtheta/2, np.pi/2, dtheta)
    assert len(thetas) == ntheta
    
    corners = ((0,        -np.pi, -np.pi), 
               (-np.pi/2, -np.pi,  np.pi),
               (-np.pi,    np.pi,  np.pi),
               (np.pi/2,   np.pi, -np.pi))
    
    kxfs  = []
    kyfs  = []
    dEdks = []
    dSdEs = []
    vels  = []
    rs    = []
    Is = interpS(SR, wr, nr, nk, izero)
    print('len(Is)', len(Is))
    for corner in corners:
        for theta in thetas:
            (kx, ky), r, dEdk =  kF(theta, corner[0], corner[1], corner[2], t, tp, mu)
            kxfs.append(kx)
            kyfs.append(ky)
            rs.append(r)
            dEdks.append(dEdk)
            v, dSdE = vel_and_dSdE(kx, ky, dEdk, Is, wr, nr, izero)
            vels.append(v)
            dSdEs.append(dSdE)
    
    # compute normalization factor
    num = 0
    dos = 0
    for ik in range(ntheta):
        dos += rs[ik] / vels[ik] * dtheta
        num += -dSdEs[ik] * rs[ik] / vels[ik] * dtheta
            
    lamb = num / dos
           
    print('done lamb')
    
    print('lamb electronic = ', lamb)


def corrected_lamb_mass(basedir, folder, ntheta=5):
    # corrected by Fermi Surface shift due to real part of Sigma
    
    wr, nr, nk, SR, DR, mu, t, tp, g0, omega = load(basedir, folder)

    izero = np.argmin(np.abs(wr))
    
    dtheta = np.pi/(2*ntheta)
    thetas = np.arange(dtheta/2, np.pi/2, dtheta)
    assert len(thetas) == ntheta
    
    corners = ((0,        -np.pi, -np.pi), 
               (-np.pi/2, -np.pi,  np.pi),
               (-np.pi,    np.pi,  np.pi),
               (np.pi/2,   np.pi, -np.pi))
    
    kxfs  = []
    kyfs  = []
    dEdks = []
    dSdEs = []
    vels  = []
    rs    = []
    Is = interpS(SR, wr, nr, nk, izero)
    print('len(Is)', len(Is))
    for corner in corners:
        for theta in thetas:
            #(kx, ky), r, dEdk =  kF(theta, corner[0], corner[1], corner[2], t, tp, mu)
            (kx, ky), r, dEdk =  corrected_kF(Is, theta, corner[0], corner[1], corner[2], t, tp, mu)
            kxfs.append(kx)
            kyfs.append(ky)
            rs.append(r)
            dEdks.append(dEdk)
            _, dSdE = vel_and_dSdE(kx, ky, dEdk, Is, wr, nr, izero)
            vels.append(np.abs(dEdk))
            dSdEs.append(dSdE)
    
    # compute normalization factor
    num = 0
    dos = 0
    for ik in range(ntheta):
        dos += rs[ik] / vels[ik] * dtheta
        num += -dSdEs[ik] * rs[ik] / vels[ik] * dtheta
            
    lamb = num / dos
           
    print('done lamb')
    
    print('lamb mass = ', lamb)
    np.save(basedir + 'data/'+folder+'/lamb_mass.npy', [lamb])

    lambk = -np.array(dSdEs)
    return lambk, np.array(rs)/np.array(vels)

def a2F(basedir, folder, ntheta=5):
    
    wr, nr, nk, SR, DR, mu, t, tp, g0, omega = load(basedir, folder)
    izero = np.argmin(np.abs(wr))
    
    dtheta = np.pi/(2*ntheta)
    thetas = np.arange(dtheta/2, np.pi/2, dtheta)
    assert len(thetas) == ntheta
    
    corners = ((-np.pi,-np.pi), 
               (np.pi, -np.pi),
               (-np.pi, np.pi),
               (np.pi, np.pi))
    
    kxfs  = []
    kyfs  = []
    dEdks = []
    vels  = []
    rs    = []
    Is = interpS(SR, wr, nr, nk, izero)
    for corner in corners:
        for theta in thetas:
            (kx, ky), r, dEdk =  kF(theta, corner[0], corner[1], t, tp, mu)
            kxfs.append(kx)
            kyfs.append(ky)
            rs.append(r)
            dEdks.append(dEdk)
            v = vel(kx, ky, dEdk, Is, wr, nr, izero)
            vels.append(v)
    
    # compute normalization factor
    dos = 0
    for ik in range(ntheta):
        dos += rs[ik] / vels[ik] * dtheta
        
        
    # get B interp
    B = -1.0/np.pi * DR.imag
    
    a2F = np.zeros(nr)
    
    for iw in range(nr//2, nr):
        print(iw, end=' ')
        ks = np.linspace(-np.pi, np.pi, nk+1)
        I = interp2d(ks, ks, B[:,:,iw], kind='linear')
        
        for ik in range(ntheta):
            a2Fk = 0
            
            for ikp in range(4*ntheta):
                dkx = kxfs[ikp] - kxfs[ik]
                dky = kyfs[ikp] - kyfs[ik]
                a2Fk += I(dkx, dky) / vels[ikp] / (2*np.pi)**2 * rs[ikp] * dtheta
                
            a2F[iw] += a2Fk / vels[ik] * rs[ik] * dtheta
            
    a2F *= g0**2 / dos
           
    print('done a2F')
    
    plt.figure()
    plt.plot(wr, a2F)
    plt.savefig(basedir + 'a2F')
    
    np.save(basedir + 'data/'+folder+'/a2F.npy', a2F)
    
    dw = (wr[-1]-wr[0]) / (len(wr)-1)
    lamb = 2 * np.sum(a2F[izero+1:] / wr[izero+1:]) * dw
    
    print('lamb_from_a2F = ', lamb)
    
    np.save(basedir + 'data/'+folder+'/lamb_from_a2F.npy', [lamb])

    
def corrected_a2F(basedir, folder, ntheta=5):
    
    wr, nr, nk, SR, DR, mu, t, tp, g0, omega = load(basedir, folder)
    izero = np.argmin(np.abs(wr))

    dtheta = np.pi/(2*ntheta)
    thetas = np.arange(dtheta/2, np.pi/2, dtheta)
    assert len(thetas) == ntheta
    
    corners = ((0,        -np.pi, -np.pi), 
               (-np.pi/2, -np.pi,  np.pi),
               (-np.pi,    np.pi,  np.pi),
               (np.pi/2,   np.pi, -np.pi))
    
    kxfs  = []
    kyfs  = []
    dEdks = []
    vels  = []
    rs    = []
    Is = interpS(SR, wr, nr, nk, izero)
    for corner in corners:
        for theta in thetas:
            #(kx, ky), r, dEdk =  kF(theta, corner[0], corner[1], corner[2], t, tp, mu)
            (kx, ky), r, dEdk =  corrected_kF(Is, theta, corner[0], corner[1], corner[2], t, tp, mu)
            kxfs.append(kx)
            kyfs.append(ky)
            rs.append(r)
            dEdks.append(dEdk)
            #v = vel(kx, ky, dEdk, Is, wr, nr)
            vels.append(np.abs(dEdk))
    
    # compute normalization factor
    dos = 0
    for ik in range(ntheta):
        dos += rs[ik] / vels[ik] * dtheta
        
    print('dos renormalized (unrenormalized version is 0.3) = ', 4 * dos / (2*np.pi)**2)

    # get B interp
    B = -1.0/np.pi * DR.imag

    izero = np.argmin(np.abs(wr))
    
    max_w = 0.3
    max_iw = np.argmin(np.abs(wr - max_w))
    
    a2F = np.zeros(nr)
    lambk = np.zeros((ntheta, nr))
    
    for iw in range(izero, max_iw):
        print(iw, end=' ')
        ks = np.linspace(-np.pi, np.pi, nk+1)
        I = interp2d(ks, ks, B[:,:,iw], kind='linear')
        
        for ik in range(ntheta):
            a2Fk = 0
            
            for ikp in range(4*ntheta):
                dkx = kxfs[ikp] - kxfs[ik]
                dky = kyfs[ikp] - kyfs[ik]
                a2Fk += I(dkx, dky) / vels[ikp] / (2*np.pi)**2 * rs[ikp] * dtheta
                
            a2F[iw] += a2Fk / vels[ik] * rs[ik] * dtheta
            
            lambk[ik, iw] = a2Fk
            
    lambk *= g0**2
    a2F *= g0**2 / dos
           
    print('done a2F')
    
    plt.figure()
    plt.plot(wr, a2F)
    plt.savefig(basedir + 'a2F single')
    
    np.save(basedir + 'data/'+folder+'/a2F.npy', a2F)
    
    dw = (wr[-1]-wr[0]) / (len(wr)-1)
    #lamb = 2 * np.sum(a2F[nr//2+1:] / wr[nr//2+1:]) * dw
    #lambk = 2 * np.sum(lambk[:,nr//2+1:] / wr[None,nr//2+1:], axis=1) * dw
    lamb = 2 * np.trapz(a2F[izero+1:] / wr[izero+1:], dx=dw)
    lambk = 2 * np.trapz(lambk[:,izero+1:] / wr[None,izero+1:], dx=dw, axis=1)
      
    print('lamb_from_a2F = ', lamb)
    
    np.save(basedir + 'data/'+folder+'/lamb_from_a2F.npy', [lamb])

    return lambk, np.array(rs)/np.array(vels)



def a2F_imag(basedir, folder, ntheta=5, separate_imag_folder=None):

    if separate_imag_folder:
        i1 = len(folder)
        for _ in range(3):
            i1 = folder.rfind('_', 0, i1-1)
        folder_ = folder[:i1]
    else:
        folder_ = folder

    params = read_params(basedir, folder_)
    t = params['t']
    tp = params['tp']
    mu = params['mu']
     
    nk = params['nk']
    g0 = params['g0']


    print('beta', params['beta'])
   

    dtheta = np.pi/(2*ntheta)
    thetas = np.arange(dtheta/2, np.pi/2, dtheta)
    assert len(thetas) == ntheta
    
    corners = ((0,        -np.pi, -np.pi), 
               (-np.pi/2, -np.pi,  np.pi),
               (-np.pi,    np.pi,  np.pi),
               (np.pi/2,   np.pi, -np.pi))
    
    kxfs  = []
    kyfs  = []
    dEdks = []
    vels  = []
    rs    = []
    for corner in corners:
        for theta in thetas:
            (kx, ky), r, dEdk = kF(theta, corner[0], corner[1], corner[2], t, tp, mu)
            kxfs.append(kx)
            kyfs.append(ky)
            rs.append(r)
            dEdks.append(dEdk)
            vels.append(np.abs(dEdk))
    
    # compute normalization factor
    dos = 0
    for ik in range(ntheta):
        dos += rs[ik] / vels[ik] * dtheta
        
    
    lamb = 0
    lambk = np.zeros(ntheta)   
    
    
    # compute D
    PI = np.load(basedir + 'data/' + folder_ + '/PI.npy')
    PI = fourier.t2w(PI, params['beta'], 2, kind='boson')

    if len(np.shape(PI))==3:
        migdal = RME2d(params, basedir)    
    elif len(np.shape(PI))==5:
        migdal = RME2dsc(params, basedir)

    vn = 2*np.arange(params['nw']+1) * np.pi / params['beta']
    D = migdal.compute_D(vn, PI)

    #D = np.load(basedir + 'data/' + folder + '/D.npy')

    # extend D

    D = np.concatenate((D, D[0,:,:][None,:,:]), axis=0)
    D = np.concatenate((D, D[:,0,:][:,None,:]), axis=1)
        
    # fourier transform....
    #beta = np.load(basedir + 'data/' + folder + '/beta.npy')[0]
    #dim  = 2
    #D    = fourier.t2w(D, beta, dim, 'boson')
    
    plt.figure()
    plt.imshow(D[:,:,0].real, origin='lower')
    plt.colorbar()
    plt.savefig(basedir + 'data/' + folder + '/Diw0')
    plt.close()
    
    
    ks = np.linspace(-np.pi, np.pi, nk+1)
    
    #I = interp2d(ks, ks, B[:,:,iw], kind='linear')
    I = interp2d(ks, ks, D[:,:,0].real, kind='linear')
    print('size of maximum real part : ', np.amax(np.real(D[:,:,0])))
    
    
    #print('size of imag part : ', np.amax(np.imag(D[:,:,0])))
    
    for ik in range(ntheta):
        a2Fk = 0
        
        for ikp in range(4*ntheta):
            dkx = kxfs[ikp] - kxfs[ik]
            dky = kyfs[ikp] - kyfs[ik]
            a2Fk += -I(dkx, dky) / vels[ikp] / (2*np.pi)**2 * rs[ikp] * dtheta
            
        lambk[ik] = a2Fk
        lamb += lambk[ik] / vels[ik] * rs[ik] * dtheta
        
    lambk *= g0**2
    lamb  *= g0**2 / dos
           
    print('lamb a2F imag', lamb)
    
    np.save(basedir + 'data/'+folder+'/lambk_a2F_imag.npy', lambk)
    np.save(basedir + 'data/'+folder+'/lamb_a2F_imag.npy', lamb)

    return lambk, np.array(rs)/np.array(vels)


def corrected_a2F_imag(basedir, folder, ntheta=5):
    
    wr, nr, nk, SR, DR, mu, t, tp, g0, omega = load(basedir, folder)
        

    dtheta = np.pi/(2*ntheta)
    thetas = np.arange(dtheta/2, np.pi/2, dtheta)
    assert len(thetas) == ntheta
    
    corners = ((0,        -np.pi, -np.pi), 
               (-np.pi/2, -np.pi,  np.pi),
               (-np.pi,    np.pi,  np.pi),
               (np.pi/2,   np.pi, -np.pi))
    
    kxfs  = []
    kyfs  = []
    dEdks = []
    vels  = []
    rs    = []
    Is = interpS(SR, wr, nr, nk, izero)
    for corner in corners:
        for theta in thetas:
            (kx, ky), r, dEdk =  corrected_kF(Is, theta, corner[0], corner[1], corner[2], t, tp, mu)
            kxfs.append(kx)
            kyfs.append(ky)
            rs.append(r)
            dEdks.append(dEdk)
            #v = vel(kx, ky, dEdk, Is, wr, nr)
            vels.append(np.abs(dEdk))
    
    # compute normalization factor
    dos = 0
    for ik in range(ntheta):
        dos += rs[ik] / vels[ik] * dtheta
        
    
    lamb = 0
    lambk = np.zeros(ntheta)   
    
    # extend D
    D = np.load(basedir + 'data/' + folder + '/D.npy')
    D = np.concatenate((D, D[0,:,:][None,:,:]), axis=0)
    D = np.concatenate((D, D[:,0,:][:,None,:]), axis=1)
    
    
    # fourier transform....
    beta = np.load(basedir + 'data/' + folder + '/beta.npy')[0]
    dim  = 2
    D    = fourier.t2w(D, beta, dim, 'boson')
    
    plt.figure()
    plt.imshow(D[:,:,0].real, origin='lower')
    plt.colorbar()
    plt.savefig(basedir + 'data/' + folder + '/Diw0')
    plt.close()
    
    
    ks = np.linspace(-np.pi, np.pi, nk+1)
    
    #I = interp2d(ks, ks, B[:,:,iw], kind='linear')
    I = interp2d(ks, ks, D[:,:,0].real, kind='linear')
    print('size of maximum real part : ', np.amax(np.real(D[:,:,0])))
    
    
    #print('size of imag part : ', np.amax(np.imag(D[:,:,0])))
    
    for ik in range(ntheta):
        a2Fk = 0
        
        for ikp in range(4*ntheta):
            dkx = kxfs[ikp] - kxfs[ik]
            dky = kyfs[ikp] - kyfs[ik]
            a2Fk += -I(dkx, dky) / vels[ikp] / (2*np.pi)**2 * rs[ikp] * dtheta
            
        lambk[ik] = a2Fk
        lamb += lambk[ik] / vels[ik] * rs[ik] * dtheta
        
    lambk *= g0**2
    lamb  *= g0**2 / dos
           
    print('lamb a2F imag', lamb)
    
    np.save(basedir + 'data/'+folder+'/lambk_a2F_imag.npy', lambk)
    np.save(basedir + 'data/'+folder+'/lamb_a2F_imag.npy', lamb)

    return lambk, np.array(rs)/np.array(vels)





def load(basedir, folder):
    
    wr = np.load(basedir+'data/'+folder+'/w.npy')
    nr = len(wr)
    nk = np.load(basedir + 'data/'+folder+'/nk.npy')[0]
    SR = np.load(basedir + 'data/'+folder+'/SR.npy')
    DR = np.load(basedir + 'data/'+folder+'/DR.npy')
    mu = np.load(basedir + 'data/'+folder+'/mu.npy')[0]
    t  = np.load(basedir + 'data/'+folder+'/t.npy')[0]
    tp = np.load(basedir + 'data/'+folder+'/tp.npy')[0]
    g0 = np.load(basedir + 'data/'+folder+'/g0.npy')[0]
    omega = np.load(basedir+'data/'+folder+'/omega.npy')[0]
    
    # extend SR and DR for interpolation
    SR = np.concatenate((SR, SR[0,...][None,...]), axis=0)
    SR = np.concatenate((SR, SR[:,0,...][:,None,...]), axis=1)
    DR = np.concatenate((DR, DR[0,:,:][None,:,:]), axis=0)
    DR = np.concatenate((DR, DR[:,0,:][:,None,:]), axis=1)
    
    
    '''
    print('SR shape', SR.shape)
    plt.figure()
    plt.plot(wr, SR[nk//4,nk//4,:,0,0].real)
    plt.plot(wr, SR[nk//4,nk//4,:,0,0].imag)
    '''
    
    return wr, nr, nk, SR, DR, mu, t, tp, g0, omega


if __name__=='__main__':
    folders = os.listdir('../data/')
    
    for i,folder in enumerate(folders):
        print(i, folder)
      
    folder = folders[0]
    
    a2F('../', folder)
    
    #compute_lamb('../', folder)
    

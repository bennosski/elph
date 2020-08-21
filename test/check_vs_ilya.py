import src
from migdal_2d import Migdal
#from renormalized_2d import Migdal
from functions import mylamb2g0, band_square_lattice, lamb2g0_ilya
from numpy import *
import os
import shutil
import sys
import time


#-------------------------------------------------
# setup

fill, omega  = 0.8, 0.17

#------------------------------------------------

#lambs = arange(0.0, 0.61, 0.05)

#lambs = linspace(0.01, 0.6, 20)
#fracs = linspace(0.8, 0.2, len(lambs))

def setup(renorm):
    # index runs from 0 to nfill-1

    params = {}
    params['nw']    = 512
    params['nk']    = 12
    params['t']     = 1.0
    params['tp']    = -0.3
    params['omega'] = omega
    params['renormalized'] = renorm
    params['sc'] = 0
    params['beta']  = 16.0
    params['band']  = band_square_lattice
    #params['g0'] = mylamb2g0(lambs[index], omega, 8.0)
    params['g0'] = None
    params['dim'] = 2
    params['Q'] = None

    #----------------------------
    params['dens'] = fill

    return params




#------------------------------------------------
# run an unrenormalized calculation for reference

params = setup(renorm=False)

basedir = '/scratch/users/bln/migdal_check_vs_ilya/single_particle_unrenorm/'
if os.path.exists(basedir):
    shutil.rmtree(basedir)

lamb = 0.4
frac = 0.8
S0, PI0 = None, None

print('2D unrenormalized Migdal')

params['g0'] = lamb2g0_ilya(lamb, omega, 8.0)
print('g0 = ', params['g0'])

time0 = time.time()
migdal = Migdal(params, basedir)

sc_iter = 2000
savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, frac=frac)

print('savedir : ', savedir)
#save(savedir+'G.npy', G)
#save(savedir+'D.npy', D)
save(savedir+'S.npy', S)
save(savedir+'GG.npy', GG)


#------------------------------------------------
# reproduce the selfenergy plots

params = setup(renorm=True)

#basedir = '/home/groups/tpd/bln/migdal_check_vs_ilya/'
basedir = '/scratch/users/bln/migdal_check_vs_ilya/single_particle/'
if os.path.exists(basedir):
    shutil.rmtree(basedir)

lambs = [0.2, 0.4, 0.5]
fracs = [0.8, 0.8, 0.2]
S0, PI0 = None, None

for i,lamb in enumerate(lambs):

    print('2D Renormalized Migdal')
    
    params['g0'] = lamb2g0_ilya(lamb, omega, 8.0)
    print('g0 = ', params['g0'])

    time0 = time.time()
    migdal = Migdal(params, basedir)

    sc_iter = 2000
    savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, frac=fracs[i])

    print('savedir : ', savedir)
    #save(savedir+'G.npy', G)
    #save(savedir+'D.npy', D)
    save(savedir+'S.npy', S)
    save(savedir+'GG.npy', GG)



#-------------------------------------------------
# run the simulation
# attempt to reproduce fig 2 (xsc, xcdw vs lamb)

params = setup(renorm=True)

#basedir = '/home/groups/tpd/bln/migdal_check_vs_ilya/'
basedir = '/scratch/users/bln/migdal_check_vs_ilya/susceptibilities/'
if os.path.exists(basedir):
    shutil.rmtree(basedir)

S0, PI0, mu0 = None, None, None
Xscs  = []
Xcdws = []

lambs = linspace(0.01, 0.6, 20)

for i,lamb in enumerate(lambs):

    print('2D Renormalized Migdal')
    
    params['g0'] = lamb2g0_ilya(lamb, omega, 8.0)

    time0 = time.time()
    migdal = Migdal(params, basedir)

    sc_iter = 2000
    savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, frac=0.2)

    if G is None:
        print('Failed to converge for lamb = ', lamb)
        exit()

    PI = params['g0']**2 * GG

    sc_iter = 2000
    Xsc, Xcdw = migdal.susceptibilities(savedir, sc_iter, G, D, GG, frac=0.8)

    if Xsc is None or Xcdw is None:
        print('Failed to converge susceptibilities for lamb = ', lamb)
        exit()
    
    Xscs.append(Xsc)
    Xcdws.append(amax(Xcdw))

    save(basedir + 'xscs_%1.1f_omega%1.6f.npy'%(fill, omega),   Xscs)
    save(basedir + 'xcdws_%1.1f_omega%1.6f.npy'%(fill, omega), Xcdws)
    save(basedir + 'lambs_%1.1f_omega%1.6f.npy'%(fill, omega), lambs[:len(Xscs)])

    print('------------------------------------------')
    print('simulation took', time.time()-time0, 's')
    print('')

    #S0, PI0 = S, PI
    S0, PI0 = None, None

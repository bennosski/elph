import src
from migdal_2d import Migdal
from functions import mylamb2g0, band_square_lattice
from numpy import *
import os
import shutil
import sys
import time

# label jobs by indices
# submit script has a job index
# job index used as input to python script to identify the correct job
# parameters are calculated based on the index

#-------------------------------------------------
# setup

print('sys.argv', sys.argv)

job_index = int(sys.argv[1])

fill, omega  = 0.8, 0.357342

#fill, omega  = 0.4, 0.2137289

#fill, omega  = 0.8, 3.57342472

#fill, omega  = 0.4, 2.137289

#------------------------------------------------

lambs = arange(0.0, 0.61, 0.05)

def setup(index):
    # index runs from 0 to nfill-1

    params = {}
    params['nw']    = 512
    params['nk']    = 8
    params['t']     = 1.0
    params['tp']    = 0.0
    params['omega'] = omega
    params['renormalized'] = True
    params['sc'] = 0
    params['Q'] = None
    params['beta']  = None
    params['band']  = band_square_lattice
    params['g0'] = mylamb2g0(lambs[index], omega, 8.0)
    params['dim'] = 2

    #----------------------------
    params['dens'] = fill

    return params

params = setup(job_index)

#-------------------------------------------------
# run the simulation

#basedir = '/scratch/users/bln/elph/imagaxis/dqmc/xs_no_initial_guess/'
#basedir = '/home/users/bln/elph/data/dqmc/xs_no_initial_guess/'

basedir = '/home/groups/tpd/bln/migdal/'

S0, PI0, mu0 = None, None, None
betas = []
Xscs  = []
Xcdws = []

#for i,lamb in enumerate(lambs):
# loop over betas

beta  = 1.6
dbeta = 0.4
while (beta < 30.0):

    print('2D Renormalized Migdal')
    
    params['beta'] = beta

    time0 = time.time()
    migdal = Migdal(params, basedir)

    sc_iter = 2000
    savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, mu0=mu0, frac=0.2)
    PI = params['g0']**2 * GG

    if G is None: break

    sc_iter = 2000
    Xsc, Xcdw = migdal.susceptibilities(sc_iter, G, D, GG, frac=0.2)

    save(savedir + 'Xsc.npy',  [Xsc])
    save(savedir + 'Xcdw.npy', [Xcdw])

    if Xsc is None or Xcdw is None: break
    
    betas.append(beta)
    Xscs.append(Xsc)
    Xcdws.append(amax(Xcdw))

    save(basedir + 'xs_single_fill_fill%1.1f_omega%1.6f_Xscs_lamb%1.6f.npy'%(fill, omega, lambs[job_index]),  Xscs)
    save(basedir + 'xs_single_fill_fill%1.1f_omega%1.6f_Xcdws_lamb%1.6f.npy'%(fill, omega, lambs[job_index]),  Xcdws)
    save(basedir + 'xs_single_fill_fill%1.1f_omega%1.6f_betas.npy', betas)

    #save(basedir + 'Xcdws_beta%1.6f.npy'%beta, Xcdws)
    #save(basedir + 'lambs.npy', lambs)

    print('------------------------------------------')
    print('simulation took', time.time()-time0, 's')
    print('')

    beta += dbeta

    mu0, S0, PI0 = mu, S, PI


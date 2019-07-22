import src
from renormalized_2d import Migdal
from params import mylamb2g0, band_square_lattice
from numpy import *
import os
from analyze import analyze_single_particle, analyze_x_vs_lamb
import shutil
import sys

# label jobs by indices?
# submit script has a job index
# job index used as input to python script to identify the correct job
# parameters are calculated based on the index

# todo
#  Tc vs omega for different lambdas....

#-------------------------------------------------
# setup

print('sys.argv', sys.argv)

job_index = int(sys.argv[1])

nlamb = 50
nfill = 51 
lambs = linspace(0.0, 0.6, nlamb)
fills = linspace(0.01, 1.0, nfill)

def setup(index):
    # index runs from 0 to nfill-1

    params = {}
    params['nw']    = 512
    params['nk']    = 8
    params['t']     = 1.0
    params['tp']    = 0.0
    params['omega'] = 0.4
    params['renormalized'] = True
    params['sc']    = 0
    params['beta']  = 4.8
    params['band']  = band_square_lattice
    params['g0'] = 0.0

    #----------------------------
    params['dens'] = fills[index]

    return params

params = setup(job_index)

#-------------------------------------------------
# run the simulation

basedir = '/scratch/users/bln/elph/imagaxis/dqmc/xs/'

S0, PI0 = None, None
fracs = linspace(0.8, 0.2, len(lambs))
Xscs  = []
Xcdws = []

# loop over lambdas for this filling
for i,lamb in enumerate(lambs):

    print('2D Renormalized Migdal')
    
    W    = 8.0
    params['g0'] = mylamb2g0(lamb, params['omega'], W)
    print('g0 is ', params['g0'])

    migdal = Migdal(params, basedir)

    sc_iter = 1200
    savedir, G, D, S, PI = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, frac=fracs[i])

    if G is None: break

    sc_iter = 400
    Xsc, Xcdw = migdal.susceptibilities(sc_iter, G, D, PI, frac=0.5)

    save(savedir + 'Xsc.npy', [Xsc])
    save(savedir + 'Xcdw.npy', [Xcdw])

    if Xsc is None or Xcdw is None: break
    
    Xscs.append(Xsc)
    Xcdws.append(amax(Xcdw))

    save(basedir + 'Xscs_fill%1.6f.npy'%params['dens'],  Xscs)
    save(basedir + 'Xcdws_fill%1.6f.npy'%params['dens'], Xcdws)
    save(basedir + 'lambs.npy', lambs)
    save(basedir + 'fills.npy', fills)

    S0, PI0 = S, PI




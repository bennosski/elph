import src
from renormalized_2d import Migdal
from params import band, lamb2g0
from numpy import *
import os
from analyze import analyze_single_particle, analyze_x_vs_lamb
import shutil

params = {}
params['nw']    = 512
params['nk']    = 12
params['t']     = 1.0
params['tp']    = -0.3
params['omega'] = 0.17
params['dens']  = 0.8
params['renormalized'] = True
params['sc']    = 1
params['beta']  = 16.0
params['g0']    = 0.125
params['band']  = band

basedir = '/scratch/users/bln/elph/imagaxis/ilya_x_vs_lamb/'
if not os.path.exists(basedir): os.mkdir(basedir)

analyze_x_vs_lamb(basedir)
exit()

shutil.rmtree(basedir + 'data/')

S0, PI0 = None, None
lambs = linspace(0.01, 0.6, 20)
fracs = linspace(0.8, 0.2, len(lambs))
Xscs  = []
Xcdws = []

for i,lamb in enumerate(lambs):

    print('2D Renormalized Migdal')
    
    W    = 8.0
    params['g0'] = lamb2g0(lamb, params['omega'], W)
    print('g0 is ', params['g0'])

    migdal = Migdal(params, basedir)

    sc_iter = 400
    savedir, G, D, S, PI = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, frac=fracs[i])

    if G is None: break

    sc_iter = 300
    Xsc, Xcdw = migdal.susceptibilities(sc_iter, G, D, PI, frac=0.7)

    save(savedir + 'Xsc.npy', [Xsc])
    save(savedir + 'Xcdw.npy', [Xcdw])

    if Xsc is None or Xcdw is None: break
    
    Xscs.append(Xsc)
    Xcdws.append(amax(Xcdw))

    save(basedir + 'Xscs.npy',  Xscs)
    save(basedir + 'Xcdws.npy', Xcdws)
    save(basedir + 'lambs.npy', lambs)

    S0, PI0 = S, PI

analyze_x_vs_lamb(basedir)


exit()

basedir = '/scratch/users/bln/elph/imagaxis/ilya/'
if not os.path.exists(basedir): os.mkdir(basedir)

lambs = [0.2, 0.4, 0.6]
S0, PI0 = None, None
fracs = [0.8, 0.8, 0.2]

for i,lamb in enumerate(lambs):

    print('2D Renormalized Migdal')
    
    W    = 8.0
    params['g0'] = lamb2g0(lamb, params['omega'], W)
    print('g0 is ', params['g0'])

    migdal = Migdal(params, basedir)

    sc_iter = 400
    savedir, G, D, S, PI = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, frac=fracs[i])
    save(savedir + 'S.npy', S)
    save(savedir + 'PI.npy', PI)
    save(savedir + 'G.npy', G)
    save(savedir + 'D.npy', D)

    #sc_iter = 300
    #Xsc, Xcdw = migdal.susceptibilities(sc_iter, G, D, PI, frac=0.7)
    #save(savedir + 'Xsc.npy',  [Xsc])
    #save(savedir + 'Xcdw.npy', [Xcdw])

    S0, PI0 = S, PI

analyze_single_particle(basedir)




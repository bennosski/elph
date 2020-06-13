import src
#from renormalized_2d import Migdal
from migdal_2d import Migdal
#from params import band, lamb2g0
from functions import band_square_lattice, lamb2g0_ilya
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
params['sc']    = 0
params['beta']  = 16.0
params['g0']    = None
#params['fixed_mu'] = -1.11
params['band']  = band_square_lattice

#-----------------------------------------------------
# to get Ilya's susceptibilities

#basedir = '/scratch/users/bln/elph/imagaxis/ilya_x_vs_lamb/'
basedir = '../../test_ilya/'
if not os.path.exists(basedir): os.mkdir(basedir)

def compute_susceptibilities(basedir):
    shutil.rmtree(basedir + 'data/')
    
    S0, PI0 = None, None
    lambs = linspace(0.01, 0.6, 20)
    fracs = linspace(0.8, 0.2, len(lambs))
    Xscs  = []
    Xcdws = []
    
    for i,lamb in enumerate(lambs):
    
        print('2D Renormalized Migdal')
        
        W    = 8.0
        params['g0'] = lamb2g0_ilya(lamb, params['omega'], W)
        print('g0 is ', params['g0'])
    
        migdal = Migdal(params, basedir)
    
        sc_iter = 400
        savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, frac=0.2)
    
        if G is None: break
    
        sc_iter = 300
        Xsc, Xcdw = migdal.susceptibilities(sc_iter, G, D, GG, frac=0.7)
    
        save(savedir + 'Xsc.npy', [Xsc])
        save(savedir + 'Xcdw.npy', [Xcdw])
    
        if Xsc is None or Xcdw is None: break
        
        Xscs.append(Xsc)
        Xcdws.append(amax(Xcdw))
    
        save(basedir + 'Xscs.npy',  Xscs)
        save(basedir + 'Xcdws.npy', Xcdws)
        save(basedir + 'lambs.npy', lambs)
    
        S0, PI0 = S, PI

#analyze_x_vs_lamb(basedir)
#exit()

#-----------------------------------------------------
# to get Ilya's single particle quantities (Sigma and PI)

def compute_single_particle(basedir):
    
    lambs = [0.2, 0.4, 0.5]
    S0, PI0 = None, None
    fracs = [0.8, 0.8, 0.2]
    
    for i,lamb in enumerate(lambs):
    
        print('2D Renormalized Migdal')
        
        W    = 8.0
        params['g0'] = lamb2g0_ilya(lamb, params['omega'], W)
        print('g0 is ', params['g0'])
    
        migdal = Migdal(params, basedir)
    
        sc_iter = 800
        savedir, mu, G, D, S, GG = migdal.selfconsistency(sc_iter, S0=S0, PI0=PI0, frac=fracs[i], cont=True)
        PI = params['g0']**2 * GG
        
        save(savedir + 'S.npy', S)
        save(savedir + 'PI.npy', PI)
        save(savedir + 'G.npy', G)
        save(savedir + 'D.npy', D)
    
        #sc_iter = 300
        #Xsc, Xcdw = migdal.susceptibilities(sc_iter, G, D, PI, frac=0.7)
        #save(savedir + 'Xsc.npy',  [Xsc])
        #save(savedir + 'Xcdw.npy', [Xcdw])
    
        S0, PI0 = S, PI


#------------------------------------------------------------

compute_single_particle(basedir)
analyze_single_particle(basedir)

#compute_susceptibilities(basedir)
#analyze_x_vs_lamb(basedir)



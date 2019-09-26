# Comparing how short the vectors that are picked using:
#
# - MCMC algorithms
# - Klein's
# - Prest's

import numpy as np
import math
import statistics as st
import GibbsUpdaters as gu
import Pickers as pi
import Functions as fn
import MetropolisHastings.KleinSampler as ks
import MetropolisHastings.MHKSampler as mhk
import MetropolisHastings.MHSampler as mh
import FALCON.sampler as fl

# A2 Basis:
a = math.sqrt(3)
Basis = np.array([[1/2, 1/2],
              [a/2, -a/2]])
#Basis = np.array([[1, 0], [0,1 ]])
q, r = np.linalg.qr(Basis, mode='reduced')
# Std dev around the smoothing parameter for A2
Var = 1.136
# Mean set to 0
Mean = np.array([0,0])
PrimeMean = np.dot(np.linalg.pinv(q), Mean)
# Cutoff is 15 std devs from the mean
Cutoff = 15

"""Gibbs"""
GibbsResults = []
#for i in range(0, 100):
    # Intial Picker
#    GibbsInit = pi.pick_inital(Basis, 2, Mean, 20, Cutoff)
    # Gibbs
#    GibbsData = gu.gibbs(Basis, 2, GibbsInit, Mean, Var, Cutoff, 500, 'RSGS')
    # Distance of last point picked from mean
#    GibbsResults.append(np.linalg.norm(GibbsData[-1] - Mean))

"""Symmetric Metropolis-Hastings"""
MHResults = []
for i in range(0, 100):
    # Initial Picker
    MHInitial = ks.KleinSampler(Basis, r, 20, PrimeMean, Cutoff)
    # Metropolis Hastings
    MHResults.append(np.linalg.norm(mh.SymMHSampler(Basis, Var, Mean, Cutoff, MHInitial[0], MHInitial[1], 500)[2]-Mean))

"""Metropolis-Hastings-Klein"""
MHKResults = []
for i in range(0, 100):
    # Initial picker
    MHKInitial = ks.KleinSampler(Basis, r, 20, PrimeMean, Cutoff)
    # Metropolis-Hastings-Klein
    MHKResults.append(np.linalg.norm(mhk.IndepMHK(Basis, Var, Mean, Cutoff, MHKInitial[0], MHKInitial[1], 500)[2] - Mean))

"""Kleins"""
KleinResults = []
for i in range(0, 100):
    KleinResults.append(np.linalg.norm(ks.KleinSampler(Basis, r, Var, PrimeMean, Cutoff) - Mean))

"""Prest's"""
PrestResults = []
for i in range(0, 100):
    PrestResults.append(np.linalg.norm(fl.sampler_z(Var, 0) -  0))

Results = {}
Results['Kleins'] = st.mean(KleinResults)
Results['Prests'] = st.mean(PrestResults)
#Results['Gibbs'] = st.mean(GibbsResults)
Results['MHK'] = st.mean(MHKResults)
Results['MH'] = st.mean(MHKResults)

for entry in Results:
    print(entry, Results[entry])

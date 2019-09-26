# Implementing the Symmetric-Metropolis-Hastings Sampler

from MetropolisHastings import MHFunctions as FN
from MetropolisHastings import KleinSampler as KS
import numpy as np
import math
import random
import statistics as st


def SymMHSampler(basis, var, mean, cutoff, InitInitial, initial, runtime, varProp = 'NA'):
    """Symmetric Metropolis-Hastings Sampler"""

    # Proposal variation given for scaling and optimisation.
    if varProp == 'NA':
        varProp = var
    Q, R = np.linalg.qr(basis, mode='reduced')
    # Xn is chain in Z^n
    Xn = [InitInitial]
    # Yn is chain in Lattice
    Yn = [initial]
    # Lag for autocorrelation
    i = 1
    lag = [0]
    # Acceptance for testing
    # acceptance = [1]
    # accRate = []
    for t in range(runtime):
        current = [Xn[t], Yn[t]]
        new_centre = np.dot(np.linalg.pinv(Q),Yn[t])
        new = KS.KleinSampler(basis, R, varProp, new_centre, cutoff)
        acc = FN.MHacceptance(basis, var, mean, current[0], new[0])
        u = random.random()
        if u <= acc:
            Xn.append(new[0])
            Yn.append(new[1])
            lag.append(i)
            # acceptance.append(acc)
            # accRate.append(1)
        else:
            Xn.append(current[0])
            Yn.append(current[1])
            lag.append(i)
            # acceptance.append(acc)
            # accRate.append(0)
        i += 1
    return [lag, Yn, Yn[-1]]

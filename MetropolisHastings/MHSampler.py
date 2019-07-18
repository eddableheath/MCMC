# Implementing the Symmetric-Metropolis-Hastings Sampler

from MetropolisHastings import MHFunctions as FN
from MetropolisHastings import KleinSampler as KS
import numpy as np
import math
import random
import statistics as st


def SymMHSampler(basis, var, mean, cutoff, InitInitial, initial, runtime):
    """Symmetric Metropolis-Hastings Sampler"""

    Q, R = np.linalg.qr(basis, mode='reduced')
    # Xn is chain in Z^n
    Xn = [InitInitial]
    # Yn is chain in Lambda
    Yn = [initial]
    # Acceptance for testing
    acceptance = [1]
    accRate = []
    for t in range(runtime):
        current = [Xn[t], Yn[t]]
        new_centre = np.dot(np.linalg.pinv(Q),Yn[t])
        new = KS.KleinSampler(basis, R, var, new_centre, cutoff)
        acc = FN.MHacceptance(basis, var, mean, current[0], new[0])
        u = random.random()
        if u <= acc:
            Xn.append(new[0])
            Yn.append(new[1])
            acceptance.append(acc)
            accRate.append(1)
        else:
            Xn.append(current[0])
            Yn.append(current[1])
            acceptance.append(acc)
            accRate.append(0)
    return [acceptance, Yn, Yn[-1], accRate]


# Testing
a = math.sqrt(3)
B = np.array([[1/2, 1/2],
              [a/2, -a/2]])
s = 1.106
c = np.array([0.5, -0.5])
L = 15

e = np.array([0, 0])

results = SymMHSampler(B, s, c, L, e, e, 100)[3]


print(st.mean(results))
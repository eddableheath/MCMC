# Implementation of the independent-Metropolis-Hastings-Klein Sampler

from MetropolisHastings import MHKFunctions as FN
from MetropolisHastings import KleinSampler as KS
import numpy as np
import math
import random
import statistics as st


def MixTime(basis, var, mean, accuracy, cutoff):
    """Pre-computing the mixing time"""

    # This is insanely computationally heavy,
    # finding the value of the gaussian at each point?!
    return


def IndepMHK(basis, var, mean, cutoff, IntInitial, initial, runtime):
    """The independent MHK lattice Gaussian sampler"""

    Q, R = np.linalg.qr(basis, mode='reduced')
    cPrime = np.dot(np.linalg.pinv(Q), mean)
    # Xn contains markov chain of v \in Z^n
    Xn = [IntInitial]
    # Yn contains markov chain of v \in \Lambda
    Yn = [initial]
    # Acceptance for testing
    acceptance = [1]
    for t in range(runtime):
        current = [Xn[t], Yn[t]]
        new = KS.KleinSampler(basis, R, var, cPrime, cutoff)
        acc = FN.acceptance(mean, var, cutoff, Q, R, current[0], new[0])
        u = random.random()
        if u <= acc:
            Xn.append(new[0])
            Yn.append(new[1])
            acceptance.append(acc)
        else:
            Xn.append(current[0])
            Yn.append(current[1])
            acceptance.append(acc)
    return [acceptance, Yn, Yn[-1]]



# Testing
a = math.sqrt(3)
B = np.array([[1/2, 1/2],
              [a/2, -a/2]])
s = 212
c = np.array([0.5, -0.5])
L = 15
m, n = np.linalg.qr(B, mode='reduced')
Primec = np.dot(np.linalg.pinv(m), c)

init = KS.KleinSampler(B, n, s, Primec, L)
results = IndepMHK(B, s, c, L, init[0], init[1], 40)[0]

print(st.mean(results))

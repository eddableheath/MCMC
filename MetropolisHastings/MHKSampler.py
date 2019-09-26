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
#B = np.array([[1/2, 1/2],
#              [a/2, -a/2]])




#print(st.mean(results))

B = np.array([[2, -1, 0, 0, 0, 0, 0, 1/2],
              [0, 1, -1, 0, 0, 0, 0, 1/2],
              [0, 0, 1, -1, 0, 0, 0, 1/2],
              [0, 0, 1, -1, 0, 0, 0, 1/2],
              [0, 0, 0, 1, -1, 0, 0, 1/2],
              [0, 0, 0, 0, 1, -1, 0, 1/2],
              [0, 0, 0, 0, 0, 1, -1, 1/2],
              [0, 0, 0, 0, 0, 0, 1, 1/2]])

UniMod = np.array([[-5, 3, -27, -43, 11, -86, -365, 1456],
                   [-2, 1, -10, -15, 4, -32, -132, 526],
                   [-1, -2, 6, 22, -4, 20, 128, -514],
                   [-5, -3, -4, 22, 0, -21, 39, -178],
                   [0, 0, 5, 13, -4, 23, 99, -387],
                   [-5, 5, -30, -50, 8, -76, -383, 1553],
                   [-2, -3, 5, 32, -7, 20, 163, -658],
                   [-5, -1, -14, -5, 3, -47, -131, 511]])

BBetter = np.matmul(UniMod, B)
s = 9.836
c = np.array([0, 0, 0, 0, 0, 0, 0, 0])
L = 15
m, n = np.linalg.qr(BBetter, mode='reduced')
Primec = np.dot(np.linalg.pinv(m), c)

init = KS.KleinSampler(BBetter, n, s, Primec, L)
results = IndepMHK(BBetter, s, c, L, init[0], init[1], 40)[0]

print(st.mean(results))
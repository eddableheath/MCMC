# Implement the Klein Sampler

import numpy as np
import math
import copy
import random as rn
import pandas as pd


def Gauss(j, var, mean=0):
    """"One dimensional Gaussian"""

    return math.exp(-(1/(2*(var**2)))*(abs(j-mean)**2))

def tilde_gen(cPrime, index, r, vec):
    """Generating the individual tilde components"""

    if index < cPrime.size:
        s = 0
        for j in range(index+1,cPrime.size):
            s += r[index][j] * vec[j]
        return (cPrime[index] -  s)/(r[index][index])
    else:
        return cPrime[index]/(r[index][index])


def OneDCutoffCheck(mean, var, cutoff, value):
    """Checking that the sum over the integers stays within the chosen cutoff"""

    lower = mean - (cutoff*var)
    upper = mean + (cutoff*var)
    if value >= lower and value <= upper:
        return True
    else:
        return False


def IntSum(mean, var, cutoff):
    """Sum of integer Gaussians"""

    total = 0
    j = round(mean)
    check = []
    while OneDCutoffCheck(mean, var, cutoff, j):
        total += Gauss(j, var, mean)
        check.append([j, Gauss(j, var, mean)])
        j += 1
    k = np.around(mean) - 1
    while OneDCutoffCheck(mean, var, cutoff, k):
        total += Gauss(k, var, mean)
        check.append([k,Gauss(k, var, mean)])
        k -= 1

    return total


def IntProbGen(mean, var, cutoff):
    """Giving the distribution for an integer lattice"""

    dist = [[0,0]]
    N = IntSum(mean, var, cutoff)
    j = round(mean)
    while OneDCutoffCheck(mean, var, cutoff, j) == True:
        Pj = Gauss(j, var, mean)/N
        dist.append([j,Pj])
        j += 1
    k = round(mean) - 1
    while OneDCutoffCheck(mean, var, cutoff, k) == True:
        Pk = Gauss(k, var, mean)/N
        dist.append([k, Pk])
        k -= 1
    del dist[0]
    distro = np.array(dist)
    return distro


def KleinSampler(basis, var, mean, cutoff):
    """Klein sampler implementation"""

    q, r = np.linalg.qr(basis, mode='reduced')
    cPrime = np.dot(np.linalg.pinv(q),mean)
    x = np.zeros(mean.size)
    for i in range(mean.size-1, 0, -1):
        i_tilde = tilde_gen(cPrime, i, r, x)
        sigma = var / r[i][i]
        probs = IntProbGen(i_tilde, sigma, cutoff)
        x[i] = np.random.choice(probs[:, 0], 1, p=probs[:, 1])
    return (x, np.dot(basis, x))


B = np.array([[1,0],[0,1]])
s = 1
c = np.array([3.5,-2.9])
L = 15

print(KleinSampler(B, s, c, L))
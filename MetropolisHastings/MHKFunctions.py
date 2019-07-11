# Want to have the possible functions that I'll be using

import numpy as np
import math
import copy
import random as rn
import pandas as pd


def TildeComp(q, r, mean, vec):
    """"Computing the scaled vector"""

    cPrime = np.dot(np.linalg.pinv(q), mean)
    tilde = []
    for i in range(0,vec.size):
        if i <= vec.size:
            s = 0
            for j in range(i+1, vec.size):
                s += r[i][j]*vec[j]
            tilde.append((cPrime[i]-s)/r[i][i])
        else:
            tilde.append(cPrime[i]/r[i][i])
    vecTilde = np.asarray(tilde)
    return vecTilde


def GaussOne(j, var, mean=0):
    '''One dimensional gaussian with optional mean'''

    return math.exp(-(1/(2*(var**2)))*(abs(j-mean)**2))


def Gauss(basis, mean, var, vec):
    '''Multivariate Gaussian function'''

    k = np.dot(basis, vec) - mean
    return math.exp(-(1/(2*(var**2)))*((np.linalg.norm(k))**2))


def OneDCutoffCheck(mean, var, cutoff, value):
    '''Checking that the sum over the integers stays within the chosen cutoff'''

    lower = mean - (cutoff*var)
    upper = mean + (cutoff*var)
    if value >= lower and value <= upper:
        return True
    else:
        return False


def IntSum(mean, var, cutoff):
    '''Sum of Gaussians over integer values'''

    total = 0
    j = round(mean)
    check = []
    while OneDCutoffCheck(mean, var, cutoff, j):
        total += GaussOne(j, var, mean)
        check.append([j, GaussOne(j, var, mean)])
        j += 1
    k = round(mean) - 1
    while OneDCutoffCheck(mean, var, cutoff, k):
        total += GaussOne(k, var, mean)
        check.append([k,GaussOne(k, var, mean)])
        k -= 1

    return total


def IntProd(mean, var, cutoff, r):
    '''Product of sum of Gaussians'''

    total = 1
    for i in range(0, mean.size):
        altvar = var / r[i][i]
        total *= IntSum(mean[i], altvar, cutoff)
    return total


def acceptance(mean, var, cutoff, q, r, oldvec, newvec):
    '''Computing the acceptance ratio'''

    oldvectilde = TildeComp(q, r, mean, oldvec)
    newvectilde = TildeComp(q, r, mean, newvec)
    num = IntProd(newvectilde, var, cutoff, r)
    denom = IntProd(oldvectilde, var, cutoff, r)
    if denom == 0:
        return 1
    else:
        return min(1, (num/denom))
















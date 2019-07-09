# Want to have the possible functions that I'll be using

import numpy as np
import math
import copy
import random as rn
import pandas as pd


def TildeComp(q, r, mean, vec):
    """"Computing the alternate vector."""

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


def OneDCutoffCheck(mean, var, cutoff, index, value):
    '''Checking that the sum over the integers stays within the chosen cutoff'''

    lower = mean[index] - (cutoff*var)
    upper = mean[index] + (cutoff*var)
    if value >= lower and value <= upper:
        return True
    else:
        return False


def IntSum(basis, mean, var, cutoff, index, tilde, r):
    '''Sum of Gaussians over integer values'''

    total = 0
    mu = np.linalg.solve(basis, mean)
    altvar = var/(r[index][index])
    j = np.around(mu)[index]
    check = []
    while OneDCutoffCheck(mu, var, cutoff, index, j):
        total += GaussOne(j, altvar, tilde[index])
        check.append([j, GaussOne(j, altvar, tilde[index])])
        j += 1
    k = np.around(mu)[index] - 1
    while OneDCutoffCheck(mu, var, cutoff, index, k):
        total += GaussOne(k, altvar, tilde[index])
        check.append([k,GaussOne(k, altvar, tilde[index])])
        k -= 1

    return total


def IntProd(basis, mean, var, cutoff, tilde, r):
    '''Product of sum of Gaussians'''

    total = 1
    for i in range(0, tilde.size):
        total *= IntSum(basis, mean, var, cutoff, i, tilde, r)
    return total


def acceptance(basis, mean, var, cutoff, q, r, oldvec, newvec):
    '''Computing the acceptance ratio'''

    oldvectilde = TildeComp(q, r, mean, oldvec)
    newvectilde = TildeComp(q, r, mean, newvec)
    num = IntProd(basis, mean, var, cutoff, newvectilde, r)
    denom = IntProd(basis, mean, var, cutoff, oldvectilde, r)
    if denom == 0:
        return 1
    else:
        return min(1, (num/denom))







E = np.array([[11,35],[5,16]])
z = np.array([6,9])
q, r = np.linalg.qr(E, mode='reduced')
c = np.array([0,0])
sigma = 1
zTilde = TildeComp(q,r,c,z)
x = np.array([1,1])
#print(zTilde)
#print(r)
#print(sigma/r[0][0])
#print((r[0][0]**2)/(2*(sigma**2)))
#print()
#print(abs(-15.0-zTilde[0]))
#print(abs(-15.0-zTilde[0])**2)
#print(-(1/(2*((sigma/r[0][0])**2))))
#print(-(1/(2*((sigma/r[0][0])**2)))*abs(-15.0-zTilde[0])**2)
#print(math.exp(-(1/(2*((sigma/r[0][0])**2)))*abs(-15.0-zTilde[0])**2))
print(acceptance(E,c,sigma,15,q,r,x,z))








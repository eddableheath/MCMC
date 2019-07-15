# Implement the Klein Sampler

import numpy as np
import math


def Gauss(j, var, mean=0):
    """"One dimensional Gaussian"""

    return math.exp(-(1/(2*(var**2)))*(abs(j-mean)**2))


def tilde_gen(cPrime, index, r, vec):
    """Generating the individual tilde components"""

    if index <= cPrime.size:
        s = 0
        for j in range(index+1, cPrime.size):
            s += r[index][j] * vec[j]
        return (cPrime[index] - s)/(r[index][index])
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
    # check = []
    while OneDCutoffCheck(mean, var, cutoff, j):
        total += Gauss(j, var, mean)
        # check.append([j, Gauss(j, var, mean)])
        j += 1
    k = np.around(mean) - 1
    while OneDCutoffCheck(mean, var, cutoff, k):
        total += Gauss(k, var, mean)
        # check.append([k,Gauss(k, var, mean)])
        k -= 1

    return total


def IntProbGen(mean, var, cutoff):
    """Giving the distribution for an integer lattice"""

    dist = [[0,0]]
    N = IntSum(mean, var, cutoff)
    j = round(mean)
    while OneDCutoffCheck(mean, var, cutoff, j) == True:
        Pj = Gauss(j, var, mean)/N
        dist.append([j, Pj])
        j += 1
    k = round(mean) - 1
    while OneDCutoffCheck(mean, var, cutoff, k) == True:
        Pk = Gauss(k, var, mean)/N
        dist.append([k, Pk])
        k -= 1
    del dist[0]
    distro = np.array(dist)
    return distro


def KleinSampler(basis, r, var, meanPrime, cutoff):
    """Klein sampler implementation"""

    x = np.zeros(meanPrime.size)
    # print(x)
    for i in range(meanPrime.size-1, -1, -1):
        i_tilde = tilde_gen(meanPrime, i, r, x)
        # print('Tilde: ', i_tilde)
        sigma = var / abs(r[i][i])
        # print('r-val: ', r[i][i])
        # print('alt-cut?: ', cutoff * abs(r[i][i]))
        # Scaling for cutoff to ensure lattice points to pick from
        alt_cut = cutoff * abs(r[i][i])
        # print('Sigma: ', sigma)
        probs = IntProbGen(i_tilde, sigma, alt_cut)
        # print('Probs: ', probs)
        x[i] = np.random.choice(probs[:, 0], 1, p=probs[:, 1])
        # print(x)
    return [x, np.dot(basis, x)]



a = math.sqrt(3)
B = np.array([[2, -1, 0, 0, 0, 0, 0, 1/2],
              [0, 1, -1, 0, 0, 0, 0, 1/2],
              [0, 0, 1, -1, 0, 0, 0, 1/2],
              [0, 0, 1, -1, 0, 0, 0, 1/2],
              [0, 0, 0, 1, -1, 0, 0, 1/2],
              [0, 0, 0, 0, 1, -1, 0, 1/2],
              [0, 0, 0, 0, 0, 1, -1, 1/2],
              [0, 0, 0, 0, 0, 0, 1, 1/2]])

D = np.array([[1/2, 1/2],
              [a/2, -a/2]])

E = np.array([[11, 35], [5, 16]])
BE = np.matmul(E, D)

C = np.array([[-9, 34], [5, -19]])
DC = np.matmul(C, D)

s = 1
c = np.array([0, 0])
L = 15


q, r = np.linalg.qr(D, mode='reduced')
#Primec = np.dot(np.linalg.pinv(q), c)
#KleinSampler(D, r, s, Primec, L)



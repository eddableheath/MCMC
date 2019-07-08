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


def create_lattice_and_indices(n_dim, extent):
    # This is a simple 1D lattice, can extend to more dimensions by simply iterating
    lattice = np.arange(-extent, extent)
    lattice = np.asarray([lattice for _ in range(n_dim)]).T
    # Creating array of indices
    indices = np.arange(0, 2 * extent)
    indices = np.asarray([indices for _ in range(n_dim)]).T
    return indices, lattice


#def transform_lattice(lattice, basis):
 #   n_dim = len(lattice)

  #  for
   # for index, point in enumerate( lattice ):


def distance(lattice, norm = 2):
    # Norm is just how you want to define your distance
    # e.g norm = 1 is 'city block' distance, moves in integer amounts
    #     norm = 2 is standard distance from a point.
    n_dim = len(lattice)
    extended_shape = ( lattice.shape[0], 1, *lattice.shape[1:] )

    print("lattice shape", lattice.shape)
    diff = lattice.reshape( *extended_shape ) - lattice
    print("difference shape", diff.shape)
    distance = (diff ** norm).sum(2)
    print("distance shape", distance.shape)

    return distance

def points_in_range(lattice, distance, r_cut):
    # Create array of zeros and then assign values in the lattice.
    points = np.zeros(lattice.shape)
    range_condition = distance < r_cut
    points[range_condition] = lattice[ range_condition ]
    # The below function isn't quite right probably but you could look up the where function to find the indices
    indices = np.where( distance[ range_condition ] )
    return indices, points


# Try and implement klein sampler here, messy but ok
def Klein(Basis, sigma, mean, q, r, vec, cutoff):
    """Klein Sampler for implementation"""

    x_tile = TildeComp(q, r, mean, vec)
    for i in range(vec.size, 0, -1):
        sigma_i = sigma / (r[i][i])











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

# Dimension and extent of the lattice
n_dim = 3
extent = 10
# Number of standard deviations for the cutoff
n_dev = 15
# Radius for the cutoff of the Gaussian
r_cut = n_dev * sigma

indices, lattice = create_lattice_and_indices(n_dim, extent)

#d = distance(lattice)
#print(d)
#print(d.shape)

# The easiest way to get all of the numbers within a range in an array is to create an array of booleans
#points_in_range = copy.copy(lattice)

#points_in_range[ lattice >  r_cut ] = 0

#print(points_in_range)
#print(lattice.shape)






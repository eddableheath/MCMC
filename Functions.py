# Sub routines to run in the main work

import random
import numpy as np
import math
import copy

# Picking a the index randomly
'''Actually kind of uncessary, but keeping this as a note for later'''


def pick_index(n):
    return random.choice(list(range(n)))


'''Computing the exponentiated sum. Based on the chose index and a given value for that index.'''


def simple_sum(basis, vector, mean, index, var_val):
    np.put(vector, [index], [var_val])
    x = np.dot(basis, vector) - mean
    total = 0
    for column in x:
        total += column ** 2
    return total


# Tail cut off check;
def tail_cutoff(basis, vector, value, index, mean, variance, cutoff_param):
    new_vec = copy.copy(vector)
    np.put(new_vec, [index], [value])
    alpha = np.dot(basis, new_vec)
    lower = (mean[index] - (cutoff_param * variance))
    upper = (mean[index] + (cutoff_param * variance))
    if alpha[index] >= lower and alpha[index] <= upper:
        return True
    else:
        return False


# Normalisation constant computation
def norm_con(basis, vector, mean, index, sd, cutoff_param):
    total = 0
    init = np.around(np.linalg.solve(basis, mean))
    ''' Ascending from the mean '''
    k = init[index]
    while tail_cutoff(basis, vector, k, index, mean, sd, cutoff_param):
        total += math.exp(-(1/(2*(sd**2)))*simple_sum(basis, vector, mean, index, k))
        k += 1
    ''' Descending from Mean '''
    j = init[index] - 1
    while tail_cutoff(basis, vector, j, index, mean, sd, cutoff_param):
        total += math.exp(-(1/(2*(sd**2)))*simple_sum(basis, vector, mean, index, j))
        j -= 1
    return total




# Generating the probabilities for each pick:
def prob_gen(basis, vector, mean, index, sd, cutoff_param):
    # compute the normalisation constant from the distribution
    N = norm_con(basis, vector, mean, index, sd, cutoff_param)
    init = np.around(np.linalg.solve(basis, mean))
    # initialising numpy array (0 vector as slightly bodge)
    dist = [[0, 0]]
    ''' Ascending from the mean '''
    k = init[index]
    while tail_cutoff(basis, vector, k, index, mean, sd, cutoff_param):
        Pk = (math.exp(-(1/(2*(sd**2)))*simple_sum(basis, vector, mean, index, k)))/N
        dist.append([k, Pk])
        k += 1
    ''' Descending from the mean '''
    j = init[index] - 1
    while tail_cutoff(basis, vector, j, index, mean, sd, cutoff_param):
        Pl = (math.exp(-(1/(2*(sd**2)))*simple_sum(basis, vector, mean, index, j)))/N
        dist.append([j, Pl])
        j -= 1
    del dist[0]
    distro = np.array(dist)
    return distro


def autocorrelate(x):
    """ Autocorrelation of a set """

    results = np.correlate(x, x, mode='full')
    return results[results.size // 2]



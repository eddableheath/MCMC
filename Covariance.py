''' Investigating the covariance matrix of the lattice '''

import numpy as np
import Functions as Fn
import Bases as B
import Pickers as P
import math



# Expectation value for specific index
def Expected(basis, index, mean, variance, cutoff_param):
    init = np.around(mean)
    N = Fn.norm_con(basis, init, mean, index, variance, cutoff_param)
    total = 0
    ''' Ascending from the mean '''
    k = round(mean[index])
    while Fn.tail_cutoff(basis, init, k, index, mean, variance, cutoff_param, 1):
        Pk = (math.exp(-(1/(2*(variance**2)))*Fn.simple_sum(basis, init, mean, index, k)))/N
        total += Pk * k
        k += 1
    ''' Descending fromt he mean '''
    l = round(mean[index]) - 1
    while Fn.tail_cutoff(basis, init, l, index, mean, variance, cutoff_param, 0):
        Pl = (math.exp(-(1/(2*(variance**2)))*Fn.simple_sum(basis, init, mean, index, l)))/N
        total += Pl * l
        l -= 1
    return total


A2 = B.B1
mu = P.pick_mean(2)
var = 1
i = Fn.pick_index(2)

print('Basis: ')
print(A2)
print('Index: ')
print(i)
print('Mean: ')
print(mu)
print('Expectation: ')
print(Expected(A2, i, mu, var, 15))

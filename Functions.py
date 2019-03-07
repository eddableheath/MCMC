# Sub routines to run in the main work

import random
import numpy as np
import math

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
def tail_cutoff(basis, vector, value, index, mean, variance, cutoff_param, direction):
    np.put(vector, [index], [value])
    alpha = np.dot(basis, vector)
    ''' direction = 1 for ascending and 0 for descending '''
    if direction == 1:
        if alpha[index] <= (mean[index] + (cutoff_param * variance)):
            return True
        else:
            return False
    elif direction == 0:
        if alpha[index] >= (mean[index] - (cutoff_param * variance)):
            return True
        else:
            return False
    else:
        print('Choose a direction, 0 - descending, 1 ascending')


# Normalisation constant computation
def norm_con(basis, vector, mean, index, sd, cutoff_param):
    total = 0
    ''' Ascending from the mean '''
    k = round(mean[index])
    while True:
        total += math.exp(-(1/(2*(sd**2)))*simple_sum(basis,vector,mean,index,k))
        k += 1
        if not tail_cutoff(basis,vector,k,index,mean,sd,cutoff_param,1):
            break
    ''' Descending from Mean '''
    l = round(mean[index]) - 1
    while True:
        total += math.exp(-(1/(2*(sd**2)))*simple_sum(basis,vector,mean,index,l))
        l -= 1
        if not tail_cutoff(basis,vector,l,index,mean,sd,cutoff_param,0):
            break
    return total



b = np.array([[1, 0],
              [0, 1]])
v = np.array([5, -7])
m = np.array([0, 0])
i = pick_index(2)

print(norm_con(b,v,m,i,1,13))

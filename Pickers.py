''' Picking the initial point and the mean '''

import numpy as np
import Functions as Fn


# Picking random mean: range [-20, 20] in all directions based on dimension.
def pick_mean(dimension):
    return np.around(np.random.uniform(-20, 20, dimension), 2)



# Picking initial point for gibbs sampler: Uniform random within a certain number of SD from mean
def pick_inital(basis, dimension, mean, var, cutoff_param):
    init = np.around(mean)
    vector = []
    for i in range(dimension):
        points = []
        ''' Ascending from mean '''
        k = round(mean[i])
        while Fn.tail_cutoff(basis, init, k, i, mean, var, cutoff_param, 1):
            points.append(k)
            k += 1
        ''' Descending from mean '''
        l = round(mean[i]) - 1
        while Fn.tail_cutoff(basis, init, l, i, mean, var, cutoff_param, 0):
            points.append(l)
            l -= 1
        vector.append(np.random.choice(points))
    return np.array(vector)




# Test data
b = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
v = np.array([4, -7, 8, -2])
m = pick_mean(4)
v = 1
cutoff = 15

#print(m)
#print(pick_iniital(b, 4, m, v, cutoff))


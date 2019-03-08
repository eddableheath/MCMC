'''Modelling the Gibbs Sampler to Measure the Autocorrelation'''

import numpy as np
import Functions as Fn

# Gibbs updater
def gibbs_update(basis,dimension,vector,mean,variance,cutoff):
    for i in range(dimension+1):
        n = Fn.pick_index(dimension)
        d = Fn.prob_gen(basis,vector,mean,n,variance,cutoff)
        ''' Picking from prob gen '''
        np.put(vector,n,np.random.choice(d[:,0],1,p=d[:,1]))
    return vector



# Test data
b = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
v = np.array([4, -7, 8, -2])
m = np.array([0.5, 0.5, 0, 3])
var = 0.4
cutoff = 15

print(gibbs_update(b,4,v,m,var,cutoff))



'''Modelling the Gibbs Sampler to Measure the Autocorrelation'''

import numpy as np
import Functions as Fn

# Gibbs updater
def gibbs_update(basis,dimension,vector,mean,variance,cutoff):
    vec = vector
    for i in range(dimension+1):
        n = Fn.pick_index(dimension)
        d = Fn.prob_gen(basis,vec,mean,n,variance,cutoff)
        ''' Picking from prob gen '''
        np.put(vec,n,np.random.choice(d[:,0],1,p=d[:,1]))
    return vec

def gibbs(basis,dimension,vector,mean,variance,cutoff,total_run):
    data = [[0,vector]]
    #for i in range(1,total_run):
     #   new_vec = gibbs_update(basis,dimension,data[i-1][1],mean,variance,cutoff)
      #  data.append([i,new_vec])
    return data[0][1]


# Test data
b = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
v = np.array([4, -7, 8, -2])
m = np.array([0.5, 0.5, 0, 3])
var = 0.4
cutoff = 15


print(gibbs(b,4,v,m,var,cutoff,5))
print(v)
gibbs_update(b,4,v,m,var,cutoff)
print(v)
gibbs_update(b,4,v,m,var,cutoff)
print(v)
print(gibbs(b,4,v,m,var,cutoff,5))
print(v)



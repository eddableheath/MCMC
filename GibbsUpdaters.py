''' Writing explicit Gibbs updating schemes '''

import copy
import numpy as np
import Functions as Fn

# Random Scan Gibbs updater (0)
def RSGS(basis,dimension,vector,mean,variance,cutoff):
    vec =copy.copy(vector)
    for i in range(dimension+1):
        n = Fn.pick_index(dimension)
        d = Fn.prob_gen(basis,vec,mean,n,variance,cutoff)
        ''' Picking from prob gen '''
        vec[n]=np.random.choice(d[:,0],1,p=d[:,1])
    return vec

# Deterministic Update Gibbs (1)
def DUGS(basis,dimension,vector,mean,variance,cutoff):
    vec =copy.copy(vector)
    for i in range(dimension):
        d = Fn.prob_gen(basis,vec,mean,i,variance,cutoff)
        ''' Picking from prob gen '''
        vec[i]=np.random.choice(d[:,0],1,p=d[:,1])
    return vec



def gibbs(basis,dimension,vector,mean,variance,cutoff,total_run,update_strategy):
    ''' Choose update strategy from list:
        1: RSGS
        2: DUGS             '''
    data = [[0,vector]]
    if update_strategy == 'RSGS':
        for i in range(1,total_run):
            new_vec = RSGS(basis,dimension,data[i-1][1],mean,variance,cutoff)
            np.ndarray.tolist(new_vec)
            data.append([i,new_vec])
    if update_strategy == 'DUGS':
        for i in range(1,total_run):
            new_vec = DUGS(basis,dimension,data[i-1][1],mean,variance,cutoff)
            np.ndarray.tolist(new_vec)
            data.append([i,new_vec])
    #gibbs_data = np.array(data)
    return data

# Test data
b = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
v = np.array([4, -7, 8, -2])
m = np.array([0.5, 0.5, 0, 3])
var = 0.4
cutoff = 15

np.ndarray.tolist(v)
print(v)

print(gibbs(b,4,v,m,var,cutoff,5,'RSGS'))
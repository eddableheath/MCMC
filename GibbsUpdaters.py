''' Writing explicit Gibbs updating schemes '''

import copy
import numpy as np
import Functions as Fn

''' Updating Schemes:
    1: RSGS
    2: DUGS
    3: REGS
    4: RPGS'''

# Random Scan Gibbs updater (RSGS)
def RSGS(basis,dimension,vector,mean,variance,cutoff):
    vec = copy.copy(vector)
    for i in range(dimension+1):
        n = Fn.pick_index(dimension)
        d = Fn.prob_gen(basis,vec,mean,n,variance,cutoff)
        ''' Picking from prob gen '''
        vec[n]=np.random.choice(d[:,0],1,p=d[:,1])
    return vec

# Deterministic Update Gibbs (DUGS)
def DUGS(basis,dimension,vector,mean,variance,cutoff):
    vec = copy.copy(vector)
    for i in range(dimension):
        d = Fn.prob_gen(basis,vec,mean,i,variance,cutoff)
        ''' Picking from prob gen '''
        vec[i]=np.random.choice(d[:,0],1,p=d[:,1])
    return vec

# REversible Gibbs (REGS)
def REGS(basis, dimension, vector, mean, variance, cutoff):
    vec = copy.copy(vector)

    ''' Forwards '''
    for i in range(dimension):
        d = Fn.prob_gen(basis, vec, mean, i, variance, cutoff)
        ''' Picking from prob gen '''
        vec[i]=np.random.choice(d[:,0],1,p=d[:, 1])

    ''' Backwards '''
    for i in range(dimension-1,0,-1):
        d = Fn.prob_gen(basis,vec,mean,i,variance,cutoff)
        ''' Picking from prob gen '''
        vec[i]=np.random.choice(d[:,0],1,p=d[:,1])
    return vec

# Random Permutation Gibbs (RPGS)
def RPGS(basis,dimension,vector,mean,variance,cutoff):
    vec = copy.copy(vector)
    perm = np.random.permutation(dimension)
    for i in range(dimension):
        d = Fn.prob_gen(basis,vec,mean,perm[i],variance,cutoff)
        ''' Picking from prob gen '''
        vec[i]=np.random.choice(d[:,0],1,p=d[:,1])
    return vec


# Gibbs sampler chain generator
def gibbs(basis,dimension,vector,mean,variance,cutoff,total_run,update_strategy):
    ''' Choose update strategy from list:
        1: RSGS
        2: DUGS
        3: REGS
        4: RPGS '''
    data = [[0,vector]]

    if update_strategy == 'RSGS':
        for i in range(1,total_run):
            new_vec = RSGS(basis,dimension,data[i-1][1],mean,variance,cutoff)
            np.ndarray.tolist(new_vec)
            data.append([i, new_vec])

    if update_strategy == 'DUGS':
        for i in range(1,total_run):
            new_vec = DUGS(basis,dimension,data[i-1][1],mean,variance,cutoff)
            np.ndarray.tolist(new_vec)
            data.append([i,new_vec])

    if update_strategy == 'REGS':
        for i in range(1,total_run):
            new_vec = REGS(basis,dimension,data[i-1][1],mean,variance,cutoff)
            np.ndarray.tolist(new_vec)
            data.append([i,new_vec])

    if update_strategy == 'RPGS':
        for i in range(1,total_run):
            new_vec = RPGS(basis,dimension,data[i-1][1],mean,variance,cutoff)
            np.ndarray.tolist(new_vec)
            data.append([i,new_vec])
    #gibbs_data = np.array(data)
    return data
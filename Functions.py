# Sub routines to run in the main work

import random
import numpy as np

# Picking a the index randomly
def pick_index(n):
    return random.choice(list(range(n)))

# Computing Kappa simplification
def kappa_sum(basis,vector,mean,index):
    x = np.delete(vector, index)
    mu = np.delete(mean, index)
    return np.sum((basis.cdot(x)-mu)**2)

# Computing normalisation coefficient
def normalisation(basis,vector,mean,index,variance):
    kappa = kappa_sum(basis,vector,mean,index)


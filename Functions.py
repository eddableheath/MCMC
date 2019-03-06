# Sub routines to run in the main work

import random
import numpy as np
from numpy import linalg as LA

# Picking a the index randomly
def pick_index(n):
    return random.choice(list(range(n)))

# Computing the exponentiated sum. Based on the chose index and a given value for that index.
def simple_sum(basis,vector,mean,index,var_val):
    new_vec = np.put(vector,[index],[var_val])
    x = np.dot(basis,new_vec) - mean
    return LA.norm(x)

# Normalisation constant computation
def norm_con(basis,vector,mean,index,sd):

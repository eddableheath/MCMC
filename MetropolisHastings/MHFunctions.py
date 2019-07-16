# Gonna need some functions for the heavy, HEAVY, computation that this SOB requires.


import numpy as np
import math


def MHacceptance(basis, var, mean, oldvec, newvec):
    a = np.dot(basis, oldvec) - mean
    b = np.dot(basis, newvec) - mean
    poly = (np.linalg.norm(a)**2) - (np.linalg.norm(b)**2)
    return min(1, math.exp(poly / (2 * (var**2))))
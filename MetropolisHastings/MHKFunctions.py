# Want to have the possible functions that I'll be using

import numpy as np
import math
import random as rn
import pandas as pd

def TildeComp(q, r, mean, vec):
    '''Computing the alternate vector.'''

    cPrime = np.dot(np.linalg.pinv(q), mean)
    tilde = []
    for i in range(0,vec.size):
        if i <= vec.size:
            s = 0
            for j in range(i+1, vec.size):
                s += r[i][j]*vec[j]
            tilde.append((cPrime[i]-s)/r[i][i])
        else:
            tilde.append(cPrime[i]/r[i][i])
    vecTilde = np.asarray(tilde)
    return vecTilde

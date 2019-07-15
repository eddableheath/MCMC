# Testing MHK sampling acceptance rate over different variances

import numpy as np
import math
from MetropolisHastings import MHKSampler as SP
from MetropolisHastings import KleinSampler as KS
import statistics as st


var = 0.1
results = [[0, 0]]
a = math.sqrt(3)
B = np.array([[1/2, 1/2],
              [a/2, -a/2]])
s = 1
c = np.array([0.5, -0.5])
L = 15
m, n = np.linalg.qr(B, mode='reduced')
Primec = np.dot(np.linalg.pinv(m), c)

init = KS.KleinSampler(B, n, s, Primec, L)

for i in range(12):
    results.append([var, st.mean(SP.IndepMHK(B, var, c, L, init[0], init[1], 40)[0])])
    var = var*2

del results[0]
print(results)
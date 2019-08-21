# Testing MHK sampling acceptance rate over different variances

import numpy as np
import math
from MetropolisHastings import MHSampler as SP
from MetropolisHastings import KleinSampler as KS
import statistics as st
import Functions as fn
import matplotlib.pyplot as plt
import Bases
import Pickers as PS


# Testing
a = math.sqrt(3)
B = np.array([[1/2, 1/2],
              [a/2, -a/2]])
s = 1.106
c = np.array([0,0])
L = 15
cpoints = np.ndarray.tolist(c)
cx = cpoints[0]
cy = cpoints[1]
e = np.array([0, 0])

B2 = np.matmul(np.array([[0, 1],
                         [-1, 5]]), B)
B3 = np.matmul(np.array([[-9, 34],
                         [5, -19]]), B)
B4 = np.matmul(np.array([[-11, -14],
                         [4, 5]]), B)
B5 = np.matmul(np.array([[0, -1],
                         [1, 3]]), B)
B6 = np.matmul(np.array([[11, 35],
                         [5, 16]]), B)
B7 = np.matmul(np.array([[-5, -13],
                        [-3, -8]]), B)
B8 = np.matmul(np.matmul(Bases.UniMod['b'], Bases.UniMod['e']), B)

results = SP.SymMHSampler(B3, s, c, L, e, e, 10000)[1]

r = 0
prelim = {}
for result in results:
    prelim[r] = result
    r += 1

x = []
y = []
ACF = {}

for entry in prelim:
    x.append(prelim[entry][0])
    y.append(prelim[entry][1])

xACF = fn.autocorrelate(x)
NormxACF = xACF / float(xACF.max())
yACF = fn.autocorrelate(y)
NormyACF = yACF / float(yACF.max())
lag = list(range(0,len(xACF)))


plt.plot(x, y, '-y')
plt.plot(cx, cy, 'ko')
plt.show()
plt.plot(lag, NormxACF, '-k')
plt.show()
plt.plot(lag, NormyACF, '-y')
plt.show()

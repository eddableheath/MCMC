# Testing MHK sampling acceptance rate over different variances

import numpy as np
import math
from MetropolisHastings import MHSampler as SP
from MetropolisHastings import KleinSampler as KS
import statistics as st
import Functions as fn
import matplotlib.pyplot as plt


# Testing
a = math.sqrt(3)
B = np.array([[1/2, 1/2],
              [a/2, -a/2]])
s = 1.106
c = np.array([4., 4.])
L = 15

e = np.array([0, 0])

results = SP.SymMHSampler(B, s, c, L, e, e, 40)[1]

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

plt.plot(lag, NormxACF, '-k')
plt.plot(lag, NormyACF, '-y')
plt.show()

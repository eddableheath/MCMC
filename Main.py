'''Modelling the Gibbs Sampler to Measure the Autocorrelation'''
import numpy as np
import Functions as fn
import GibbsUpdaters as gu
import Pickers as pi
import math
import matplotlib.pyplot as plt


# Test data, A2 basis
#rt3 = math.sqrt(3)
b = np.array([[1/2, math.sqrt(3)/2],
              [1/2, -math.sqrt(3)/2]])
balt = np.array([[1, 0],
                 [0, 1]])
var = 1
cutoff = 15
m = pi.pick_mean(2)
v = pi.pick_inital(b, 2, m, var, cutoff)

# print(m)
# print(v)
# print(gu.gibbs(b, 2, v, m, var, cutoff, 5, 'DUGS'))

newm = np.ndarray.tolist(m)

results = []
gibbs_data = gu.gibbs(b, 2, v, m, var, cutoff, 1000, 'RSGS')
x_results = []
y_results = []

#print(gibbs_data)

# Formatting data for autocorrelation results
for datapoint in gibbs_data:
    x_results.append(datapoint[1][0])
    y_results.append(datapoint[1][1])

xACF = fn.autocorrelate(x_results)
yACF = fn.autocorrelate(y_results)
lag = list(range(0,len(xACF)))
NormxACF = xACF / float(xACF.max())
NormyACF = yACF / float(yACF.max())

#print(NormxACF)
#print(NormyACF)
#print(x_results)
#print(y_results)



plt.plot(lag, NormxACF, '-k')
plt.ylabel('Autocorrelation')
plt.xlabel('Lag')
plt.show()
plt.plot(lag, NormyACF, '-y')
plt.ylabel('Autocorrelation')
plt.xlabel('Lag')
plt.show()
plt.plot(x_results, y_results, '-y')
plt.plot(newm[0], newm[1], 'ko')
plt.ylabel('y')
plt.xlabel('x')
plt.show()


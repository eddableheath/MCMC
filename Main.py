'''Modelling the Gibbs Sampler to Measure the Autocorrelation'''
import numpy as np
import Functions as fn
import GibbsUpdaters as gu
import Pickers as pi
import math
import copy


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


results = []
gibbs_data = gu.gibbs(b, 2, v, m, var, cutoff, 5, 'DUGS')
x_results = []
y_results = []


# Formatting data for autocorrelation results
for datapoint in gibbs_data:
    x_results.append(datapoint[1][0])
    y_results.append(datapoint[1][1])
    good_x = copy.copy(x_results)
    good_y = copy.copy(y_results)
    results.append([datapoint[0], good_x, good_y])

# final data
final_results = []
for result in results:
    final_results.append([result[0],
                         fn.autocorrelate(result[1]),
                         fn.autocorrelate(result[2])])

print(final_results)
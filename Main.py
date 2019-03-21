'''Modelling the Gibbs Sampler to Measure the Autocorrelation'''
import numpy as np
import Functions as Fn
import GibbsUpdaters as GU
import Bases as B
import Pickers as Pi


# Test data
basis = B.B3
dim = 2
mu = Pi.pick_mean(2)
var = 1
init = Pi.pick_inital(basis, 2, mu, var, 15)

print('Mean:')
print(mu)
print('Initial vector:')
print(init)
#print('Gibbs: ')
#print(GU.gibbs(B.B3, 2, init, mu, var, 15, 10, 'RSGS'))
print('test:')
a1 = GU.DUGS(basis, dim, init, mu, var, 15)
print(a1)
a2 = GU.DUGS(basis, dim, a1, mu, var, 15)
print(a2)
a3 = GU.DUGS(basis, dim, a2, mu, var, 15)
print(a3)
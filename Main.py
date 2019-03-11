'''Modelling the Gibbs Sampler to Measure the Autocorrelation'''



# Test data
b = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
v = np.array([4, -7, 8, -2])
m = np.array([0.5, 0.5, 0, 3])
var = 0.4
cutoff = 15

np.ndarray.tolist(v)
print(v)

print(gibbs(b,4,v,m,var,cutoff,5))



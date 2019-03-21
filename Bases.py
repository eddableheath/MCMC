''' Library of Bases '''

# All bases for the lattice A2 with some unimodular transform applied

import numpy as np

# To simplify
rootthree = np.sqrt(3)

# Standard basis for A2
B1 = np.array([[1/2, rootthree],
               [1/2, -rootthree]])

# Library of unimodular matrices
UniMod = {'a': np.array([[0, 1],
                       [-1, 5]]),
          'b': np.array([[-9, 34],
                       [5, -19]]),
          'c': np.array([[-11, -14],
                       [4, 5]]),
          'd': np.array([[0, -1],
                       [1, 3]]),
          'e': np.array([[11, 35],
                         [5, 16]]),
          'f': np.array([[-5, -13],
                         [-3, -8]])}

B2 = np.matmul(UniMod['a'], B1)
B3 = np.matmul(UniMod['b'], B1)
B4 = np.matmul(UniMod['c'], B1)
B5 = np.matmul(UniMod['d'], B1)
B6 = np.matmul(UniMod['e'], B1)
B7 = np.matmul(UniMod['f'], B1)
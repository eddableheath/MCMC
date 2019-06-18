# Gonna need some functions for the heavy, HEAVY, computation that this SOB requires.


import numpy as np
import pandas as pd
import math

rootthree = np.sqrt(3)

B1 = np.array([[1/2, rootthree],
               [1/2, -rootthree]])

q, r = np.linalg.qr(B1, mode='reduced')

print(q)

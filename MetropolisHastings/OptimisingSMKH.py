# Optimising the Symmetric Metropolis Hastings Klein by scaling the proposal variance
# In theory optimal acceptance rate is around 0.234, can attempt to scaling that.

import numpy as np
import math as mt
import statistics as st
import MetropolisHastings.MHSampler as smhk

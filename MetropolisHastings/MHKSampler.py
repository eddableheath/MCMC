# Implementation of the independent-Metropolis-Hastings-Klein Sampler

from MetropolisHastings import MHKFunctions as FN
from MetropolisHastings import KleinSampler as KS
import numpy as np
import math


def MixTime(basis, var, mean, accuracy, cutoff):
    """Pre-computing the mixing time"""

    # This is insanely computationally heavy,
    # finding the value of the gaussian at each point?!
    return

def IndepMHK(Basis, var, mean, cutoff, )
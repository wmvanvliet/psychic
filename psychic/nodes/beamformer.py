from basenode import BaseNode
from ..dataset import DataSet
import scipy
import numpy as np
from psychic.nodes.spatialfilter import (SpatialFilter, sfilter_trial,
        plain_cov0, trial_cov0)

class SpatialBeamformer(SpatialFilter):
    '''
    A beamformer that behaves like a spatial filter.

    Parameters
    ----------
    L : 2D array (sources x channels)
        The leadfield matrix of the signals to isolate
    '''
    def __init__(self, L):
        SpatialFilter.__init__(self, None)
        self.L = L

    def train_(self, d):
        if d.data.ndim == 2:
            self.cov_x = plain_cov0(d)
        elif d.data.ndim == 3:
            self.cov_x = trial_cov0(d)
        else:
            raise ValueError('Cannot operate on %d-dimensional data.' % d.data.ndim)

        self.cov_s = np.cov(self.L, rowvar=0)
        V, self.W = scipy.linalg.eig(self.cov_s, self.cov_x)
        print np.real(V)
        self.W = np.real(self.W)
        self.W = self.W[:, np.argsort(V)]
        self.W = self.W.T[:, :self.L.shape[0]]

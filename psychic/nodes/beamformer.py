from basenode import BaseNode
from ..dataset import DataSet
import numpy as np
from ..stat import lw_cov
from psychic.nodes.spatialfilter import SpatialFilter


class SpatialBeamformer(SpatialFilter):
    '''
    Spatial beamformer.

    Parameters
    ----------
    template : 1D array (n_channels)
       Spatial activation pattern of the component to extract.

    reg : float (default: 0.2)
        Regularization parameter for the covariance matrix inversion. Also
        known as diagonal loading.

    cov_func : function (default: lw_cov)
        Covariance function to use. Defaults to Ledoit & Wolf's function.
    '''
    def __init__(self, template, reg=0.2, cov_func=lw_cov, lcmv=False):
        SpatialFilter.__init__(self, 1)
        self.template = template
        self.reg = reg
        self.cov_func = cov_func
        self.template = np.asarray(template).flatten()[:, np.newaxis]
        self.lcmv = lcmv

    def center(self, d):
        """Center the data in time"""
        return d

    def train_(self, d):
        d = self.center(d)

        # Calculate spatial covariance matrix
        sigma_x = np.mean(
            [self.cov_func(t) for t in np.rollaxis(d.data, -1)],
            axis=0)
        sigma_x += self.reg * np.eye(sigma_x.shape[0])
        sigma_x_i = np.linalg.inv(sigma_x)
        self.W = sigma_x_i.dot(self.template)
        if self.lcmv:
            self.W = self.W.dot(
                reduce(np.dot, self.template.T, sigma_x_i, self.template)
            )

    def apply_(self, d):
        d = self.center(d)
        return SpatialFilter.apply_(self, d)


class TemplateBeamformer(BaseNode):
    '''
    Beamformer operating on a template.

    Parameters
    ----------
    template : 2D array (n_channels, n_samples)
       Activation pattern of the component to extract.

    reg : float (default: 0.2)
        Regularization parameter for the covariance matrix inversion. Also
        known as diagonal loading.

    cov_func : function (default: lw_cov)
        Covariance function to use. Defaults to Ledoit & Wolf's function.
    '''
    def __init__(self, template, reg=0.2, cov_func=lw_cov, lcmv=False):
        BaseNode.__init__(self)
        self.template = template
        self.reg = reg
        self.cov_func = cov_func
        self.template = np.atleast_2d(template)
        self.lcmv = lcmv

    def center(self, d):
        """Center the data in time"""
        return d

    def train_(self, d):
        d = self.center(d)

        nsamples, ntrials = d.data.shape[1:]
        template = self.template[:, :nsamples]
        sigma_x = self.cov_func(d.data.reshape(-1, ntrials))
        sigma_x += self.reg * np.eye(sigma_x.shape[0])
        sigma_x_i = np.linalg.inv(sigma_x)

        template = self.template.flatten()[:, np.newaxis]
        self.W = sigma_x_i.dot(template).ravel()

        if self.lcmv:
            self.W = self.W.dot(
                reduce(np.dot, self.template.T, sigma_x_i, self.template)
            )

    def apply_(self, d):
        d = self.center(d)

        ntrials = d.data.shape[2]
        X = self.W.dot(d.data.reshape(-1, ntrials))
        return DataSet(data=X, feat_lab=None, default=d)

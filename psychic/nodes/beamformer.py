import numpy as np
from sklearn.covariance import (EmpiricalCovariance, ShrunkCovariance, OAS,
                                LedoitWolf)

from basenode import BaseNode
from ..dataset import DataSet
from ..trials import baseline, concatenate_trials
from spatialfilter import SpatialFilter

class SpatialBeamformer(SpatialFilter):
    '''
    LCMV Spatial beamformer.

    Parameters
    ----------
    template : 1D array (n_channels)
        Spatial activation pattern of the component to extract.

    shrinkage : str | float (default: 'oas')
        Shrinkage parameter for the covariance matrix inversion. This can
        either be speficied as a number between 0 and 1, or as a string
        indicating which automated estimation method to use:

        'none': No shrinkage: emperical covariance
        'oas': Oracle approximation shrinkage
        'lw': Ledoit-Wolf approximation shrinkage

    center : bool (default: True)
        Whether to remove the channel mean before applying the filter.
    '''
    def __init__(self, template, shrinkage='oas', center=True):
        SpatialFilter.__init__(self, 1)
        self.template = template
        self.template = np.asarray(template).flatten()[:, np.newaxis]
        self.center = center

        if center:
            self.template -= self.template.mean()

        if shrinkage == 'oas':
            self.cov = OAS
        elif shrinkage == 'lw':
            self.cov = LedoitWolf
        elif shrinkage == 'none':
            self.cov = EmpiricalCovariance
        elif type(shrinkage) == float or type(shrinkage) == int:
            self.cov = ShrunkCovariance(shrinkage=shrinkage)

    def train_(self, d):
        if self.center:
            d = DataSet(d.data - d.data.mean(axis=0), default=d)
            d = baseline(d)

        # Calculate spatial covariance matrix
        c = self.cov.fit(concatenate_trials(d).X)
        sigma_x_i = c.precision_

        # Compute spatial LCMV filter
        self.W = sigma_x_i.dot(self.template)

        # Noise normalization
        self.W = self.W.dot(
            np.linalg.inv(
                reduce(np.dot, [self.template.T, sigma_x_i, self.template])
            )
        )

    def apply_(self, d):
        if self.center:
            d = DataSet(d.data - d.data.mean(axis=0), default=d)
            d = baseline(d)
        return SpatialFilter.apply_(self, d)


class TemplateBeamformer(BaseNode):
    '''
    Spatio-temporal LCMV beamformer operating on a spatio-temporal template.

    Parameters
    ----------
    template : 2D array (n_channels, n_samples)
       Spatio-temporal activation pattern of the component to extract.

    shrinkage : str | float (default: 'oas')
        Shrinkage parameter for the covariance matrix inversion. This can
        either be speficied as a number between 0 and 1, or as a string
        indicating which automated estimation method to use:

        'none': No shrinkage: emperical covariance
        'oas': Oracle approximation shrinkage
        'lw': Ledoit-Wolf approximation shrinkage

    center : bool (default: True)
        Whether to remove the data mean before applying the filter.
    '''
    def __init__(self, template, shrinkage='oas', center=True):
        BaseNode.__init__(self)
        self.template = template
        self.template = np.atleast_2d(template)
        self.center = center

        if center:
            self.template -= self.template.mean()

        if shrinkage == 'oas':
            self.cov = OAS
        elif shrinkage == 'lw':
            self.cov = LedoitWolf
        elif shrinkage == 'none':
            self.cov = EmpiricalCovariance
        elif type(shrinkage) == float or type(shrinkage) == int:
            self.cov = ShrunkCovariance(shrinkage=shrinkage)

    def _center(self, d):
        data_mean = d.data.reshape(-1, len(d)).mean(axis=1)
        data_mean = data_mean.reshape(d.feat_shape + (1,))
        return DataSet(d.data - data_mean, default=d)

    def train_(self, d):
        if self.center:
            d = self._center(d)

        nsamples, ntrials = d.data.shape[1:]
        template = self.template[:, :nsamples]

        c = self.cov.fit(d.data.reshape(-1, ntrials).T)
        sigma_x_i = c.precision_

        template = self.template.flatten()[:, np.newaxis]
        self.W = sigma_x_i.dot(template)

        # Noise normalization
        self.W = self.W.dot(
            np.linalg.inv(reduce(np.dot, [template.T, sigma_x_i, template]))
        )

    def apply_(self, d):
        if self.center:
            d = self._center(d)

        ntrials = d.data.shape[2]
        X = self.W.T.dot(d.data.reshape(-1, ntrials))
        return DataSet(data=X, feat_lab=None, default=d)

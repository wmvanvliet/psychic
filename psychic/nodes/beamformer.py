import numpy as np
from psychic.nodes.spatialfilter import BaseSpatialFilter, sfilter_trial
import psychic


def _calc_beamformer_snr(X1, X2, nc=1, a=1.0):
        nsamples, nchannels, ninstances1 = X1.shape
        ninstances2 = X2.shape[2]

        R1 = np.zeros((nchannels, nchannels, ninstances1))
        for i in range(ninstances1):
            cov = X1[...,i].T.dot(X1[...,i])
            R1[...,i] = cov / np.trace(cov)
        R1 = np.mean(R1, axis=2)

        R2 = np.zeros((nchannels, nchannels, ninstances2))
        for i in range(ninstances2):
            cov = X2[...,i].T.dot(X2[...,i])
            R2[...,i] = cov / np.trace(cov)
        R2 = np.mean(R2, axis=2)

        V, W = np.linalg.eig(np.linalg.pinv(a*R2).dot(R1))
        order = np.argsort(V)[::-1]
        V = V[order][:nc]
        W = np.real(W[:,order][:,:nc])

        return (V,W)

def _calc_beamformer_fc(Xs, nc=1, theta=1.0):
        nsamples, nchannels = Xs[0].shape[:2]
        ninstances = [X.shape[2] for X in Xs]
        p = [n / float(np.sum(ninstances)) for n in ninstances]

        means = [np.mean(X, axis=2) for X in Xs]
        grand_avg = np.mean(np.concatenate([m[..., np.newaxis] for m in means], axis=2), axis=2)

        S_b = np.zeros((nchannels, nchannels))
        for i in range(len(Xs)):
            diff = means[i] - grand_avg
            S_b += p[i] * diff.T.dot(diff)

        S_w = np.zeros((nchannels, nchannels))
        for i,X in enumerate(Xs):
            for k in range(ninstances[i]):
                diff = X[...,k] - means[i]
                S_w += diff.T.dot(diff)

        I = np.identity(nchannels)

        V, W = np.linalg.eig(np.linalg.pinv((I-theta).dot(S_w) + theta*I).dot(S_b))

        order = np.argsort(V)[::-1]
        V = V[order][:nc]
        W = np.real(W[:,order][:,:nc])

        return (V,W)


class BeamformerSNR(BaseSpatialFilter):
    def __init__(self, nc=1, a=1.0):
        assert 0 < a <= 1, 'Regularization parameter should be in range (0; 1]'
        BaseSpatialFilter.__init__(self, 1)
        self.nc = nc
        self.a = a

    def train_(self, d):
        assert len(d.feat_shape) == 2, 'Expecting sliced data'
        assert d.nclasses == 2, 'Expecting exactly two classes'

        X1 = d.get_class(0).ndX
        X2 = d.get_class(1).ndX
        self.V, self.W = _calc_beamformer_snr(X1, X2, self.nc, self.a)

class BeamformerFC(BaseSpatialFilter):
    def __init__(self, nc=1, theta=1.0):
        assert 0 < theta <= 1, 'Regularization parameter should be in range (0; 1]'
        BaseSpatialFilter.__init__(self, 1)
        self.nc = nc
        self.theta = theta

    def train_(self, d):
        assert len(d.feat_shape) == 2, 'Expecting sliced data'
        assert d.nclasses > 1, 'Expecting more than one class'

        Xs = [d.get_class(i).ndX for i in range(d.nclasses)]
        self.V, self.W = _calc_beamformer_fc(Xs, self.nc, self.theta)

class BeamformerCFMS(BaseSpatialFilter):
    def __init__(self, nc=2, theta=1.0):
        assert nc % 2 == 0, 'Number of components should be even'
        assert 0 < theta <= 1, 'Regularization parameter should be in range (0; 1]'
        BaseSpatialFilter.__init__(self, 1)
        self.nc = nc
        self.theta = theta

    def train_(self, d):
        assert len(d.feat_shape) == 2, 'Expecting sliced data'
        assert d.nclasses == 2, 'Expecting exactly two classes'
        nchannels = d.feat_shape[1]

        # Calculate FC beamformer
        X1 = d.get_class(0).ndX
        X2 = d.get_class(1).ndX
        W_fc,_ = _calc_beamformer_fc([X1, X2], nchannels, self.theta)

        # Apply FC beamformer
        d2 = sfilter_trial(d, W_fc)

        # Calculate SNR beamformer on the result (skip first nc/2 components)
        X1 = d2.get_class(0).ndX[:,self.nc/2:,:]
        X2 = d2.get_class(1).ndX[:,self.nc/2:,:]
        W_snr,_ = _calc_beamformer_snr(X1, X2, self.nc/2, self.theta)

        print W_fc.shape
        print W_snr.shape
        self.W = np.dot(W_fc, W_snr)

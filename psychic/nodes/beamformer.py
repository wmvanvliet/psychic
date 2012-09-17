'''
Implements beamformer spatial filters to separate ERPs as described in [1].

[1] Gabriel Pires, Urbano Nunes, and Miguel Castelo-Brance. Statistical spatial
filtering for a P300-based BCI: Test in able-bodied, and patient with cerebral
palsy and amyotrophic lateral sclerosis. Journal of Neuroscience Methods, 
195:270-281, 2011.
'''

import numpy as np
from psychic.nodes.spatialfilter import BaseSpatialFilter, sfilter_trial
import scipy

def _calc_beamformer_snr(X1, X2, nc=1, a=1.0):
        nchannels, nsamples, ninstances1 = X1.shape
        ninstances2 = X2.shape[2]

        R1 = np.zeros((nchannels, nchannels, ninstances1))
        for i in range(ninstances1):
            cov = X1[...,i].dot(X1[...,i].T)
            R1[...,i] = cov / np.trace(cov)
        R1 = np.mean(R1, axis=2)

        R2 = np.zeros((nchannels, nchannels, ninstances2))
        for i in range(ninstances2):
            cov = X2[...,i].dot(X2[...,i].T)
            R2[...,i] = cov / np.trace(cov)
        R2 = np.mean(R2, axis=2)

        #V, W = np.linalg.eig(np.linalg.pinv(a*R2).dot(R1))
        V, W = scipy.linalg.eig(R1, (R1 + a*R2))
       
        order = np.argsort(V)[::-1]
        V = V[order][:nc]
        W = np.real(W[:,order][:,:nc])

        return (V,W)

def _calc_beamformer_fc(Xs, nc=1, theta=1.0):
        nchannels, nsamples = Xs[0].shape[:2]
        ninstances = [X.shape[2] for X in Xs]
        p = [n / float(np.sum(ninstances)) for n in ninstances]

        means = [np.mean(X, axis=2) for X in Xs]
        grand_avg = np.mean(np.concatenate([m[..., np.newaxis] for m in means], axis=2), axis=2)

        S_b = np.zeros((nchannels, nchannels))
        for i in range(len(Xs)):
            diff = means[i] - grand_avg
            S_b += p[i] * diff.dot(diff.T)

        S_w = np.zeros((nchannels, nchannels))
        for i,X in enumerate(Xs):
            for k in range(ninstances[i]):
                diff = X[...,k] - means[i]
                S_w += diff.dot(diff.T)

        I = np.identity(nchannels)

        #V, W = np.linalg.eig(np.linalg.pinv((I-theta).dot(S_w) + theta*I).dot(S_b))
        V, W = scipy.linalg.eig(S_b, (I-theta).dot(S_w) + theta*I)

        order = np.argsort(V)[::-1]
        V = V[order][:nc]
        W = np.real(W[:,order][:,:nc])

        return (V,W)


class BeamformerSNR(BaseSpatialFilter):
    '''
    A spatial ERP filter that aims to separate two classes. It optimizes the
    signal to noise ratio where class 0 is taken as signal and class 1 as noise.
    '''
    def __init__(self, nc=1, theta=1.0):
        '''
        Creates an max-SNR beamformer that will generate nc components. A
        regularization parameter theta can be supplied (0..1) to prevent
        overfitting.
        '''
        assert 0 < theta <= 1, 'Regularization parameter should be in range (0; 1]'
        BaseSpatialFilter.__init__(self, 1)
        self.nc = nc
        self.theta = theta

    def train_(self, d):
        assert len(d.feat_shape) == 2, 'Expecting sliced data'
        assert d.nclasses == 2, 'Expecting exactly two classes'

        X1 = d.get_class(0).ndX
        X2 = d.get_class(1).ndX
        self.V, self.W = _calc_beamformer_snr(X1, X2, self.nc, self.theta)

class BeamformerFC(BaseSpatialFilter):
    '''
    A spatial ERP filter that aims to separate two classes. It optimizes the
    Fisher's criterion, so it increases the separation between classes while
    minimizing the variance within a class.
    '''
    def __init__(self, nc=1, theta=1.0):
        '''
        Creates an FC beamformer that will generate nc components. A
        regularization parameter theta can be supplied (0..1) to prevent
        overfitting.
        '''
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
    '''
    A combination of the BeamformerFC and BeamformerSNR classes that combines
    both methods in a suboptimum way. First BeamformerFC is run to generate the
    first nc/2 components. Then, BeamformerSNR is run on the FC result to
    generate the last nc/2 components.
    '''
    def __init__(self, nc=2, theta=1.0):
        '''
        Creates an CFMS beamformer that will generate nc components, where half
        of the components are supplied by an FC beamformer and half are supplied
        by an max-SNR beamformer. NC should therefore alsways be an even number.
        A regularization parameter theta can be supplied (0..1) to prevent
        overfitting. 
        '''
        assert nc % 2 == 0, 'Number of components should be even'
        assert 0 < theta <= 1, 'Regularization parameter should be in range (0; 1]'
        BaseSpatialFilter.__init__(self, 1)
        self.nc = nc
        self.theta = theta

    def train_(self, d):
        assert len(d.feat_shape) == 2, 'Expecting sliced data'
        assert d.nclasses == 2, 'Expecting exactly two classes'
        nchannels = d.feat_shape[0]

        # Calculate FC beamformer
        X1 = d.get_class(0).ndX
        X2 = d.get_class(1).ndX
        _, W_fc = _calc_beamformer_fc([X1, X2], nchannels, self.theta)

        # Apply FC beamformer
        d2 = sfilter_trial(d, W_fc)

        # Calculate SNR beamformer on the result (skip first nc/2 components)
        X1 = d2.get_class(0).ndX[:,self.nc/2:,:]
        X2 = d2.get_class(1).ndX[:,self.nc/2:,:]
        _, W_snr = _calc_beamformer_snr(X1, X2, self.nc/2, self.theta)

        I = np.identity(nchannels)

        # Construct filter that will take the first nc/2 FC components applied
        # to the EEG data, and then the first nc/2 SNR components applied to the
        # FC filtered data.
        self.W = np.dot(W_fc, np.concatenate((I[:,:self.nc/2], W_snr), axis=1))

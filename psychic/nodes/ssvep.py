# coding=utf-8

import numpy as np
from golem import DataSet
from golem.nodes import BaseNode
import itertools
import spectrum

class SLIC(BaseNode) :
    '''
    Node that applies the SLIC [1] algorithm to the data. The algorithm
    receives a number of frequencies as input. For each frequency, it splits
    the data in segments of 1/freq length and computes the correlation between
    each segment and the average. Signals with a high periodicity of the given
    frequency (such as SSVEP signals) will have a high correlation value,
    signals with a low periodicity or a periodicity with a different frequency
    will have a low correlation.

    Expected input:
    instances: trials/windows
    features: sampels*channels

    Output:
    instances: samples (one for each window)
    features: channels (one frequency per channel)

    [1] Manyakov, Nikolay V, Nikolay Chumerin, Adrien Combaz, Arne Robben, and
    Marc M Van Hulle. 2010.  "Decoding SSVEP Responses Using Time Domain
    Classification." in International Conference on Neural Computation.
    Valentia, Spain.
    '''

    def __init__(self, sample_rate, frequencies):
        '''
        Create a new SLIC node.

        Required parameters:
        sample_rate - The sample rate of the data. Needed, because the
                      get_samplerate() utility function of psychic only works
                      before windowing of the data.
        frequencies - A list of frequencies to test for
        '''
        BaseNode.__init__(self)
        self.sample_rate = sample_rate
        self.frequencies = frequencies
        self.nfrequencies = len(frequencies)

    def train_(self, d):
        pass

    def apply_(self, d):
        if d.ninstances == 0:
            return DataSet(default=d)

        mean_corrs = self.slic(d)

        X = np.max(mean_corrs, axis=1)
        feat_shape = X.shape[:-1]
        feat_lab = ['%d Hz' % f for f in self.frequencies] 

        return DataSet(X=X, feat_shape=feat_shape, feat_lab=feat_lab, default=d)

    def slic(self, d):
        nchannels, nsamples, ntrials = d.ndX.shape

        mean_corrs = np.zeros( (self.nfrequencies, nchannels, ntrials) )
        for f,freq in enumerate(self.frequencies):

            # Determine the number and length of the periods
            period_length = self.sample_rate / float(freq)
            #nperiods = int(nsamples/period_length)

            # Construct the period indices
            offsets = np.round( np.arange(period_length, nsamples, period_length) - period_length )
            length = np.arange( int(period_length) )

            x,y = np.meshgrid(offsets, length)
            period_idxs = (x+y).astype(np.int).T

            for trial in range(ntrials):
                # Split the signal into periods
                periods = d.ndX[:,period_idxs,trial]

                # Calculate the mean of the periods
                avg = np.mean(periods, axis=1)

                # Calculate the mean correlation between each period and the mean
                for channel in range(nchannels):
                    corrs = []
                    for period in range(0, periods.shape[1]):
                        corr = np.corrcoef(avg[channel,:], periods[channel,period,:])
                        corrs.append( 0.5*(corr + 1) )
                    mean_corrs[f, channel, trial] = np.mean(corrs)

        return mean_corrs

class SSVEPNoiseReduce(BaseNode):
    '''
    Node that tries to reduce EEG activity that is not informative for SSVEP
    classification using the method described in:

    Friman, O., Volosyak, I., & Gräser, A. (2007). Multiple channel detection
    of steady-state visual evoked potentials for brain-computer interfaces.
    IEEE transactions on bio-medical engineering, 54(4), 742–50.
    '''
    def __init__(self, sample_rate, frequencies, nharmonics=2, retain=0.1, nsamples=None):
        '''
        parameters:
            sample_rate - sample rate of the signal
            frequencies - SSVEP frequencies that should be optimized for
            nharmonics  - number of harmonics of the SSVEP frequencies to optimize for
            retain      - amount of noise variation to retain [0 - 1.0]
            nsamples    - number of samples in the EEG trials, when specified,
                          training of the node can be done without training data
        '''
        BaseNode.__init__(self)
        self.sample_rate = sample_rate
        self.frequencies = frequencies
        self.retain = retain
        self.harmonics = np.arange(nharmonics+1) + 1
        self.nharmonics = len(self.harmonics)
        self.nsamples = nsamples

    def train_(self, d):
        if self.nsamples == None:
            assert d != None, 'Cannot determine window length.'
            self.nsamples = d.ndX.shape[1]

        # Construct a list of the SSVEP frequencies plus their harmonics
        freqs_to_remove = [f*h for f,h in itertools.product(self.frequencies, self.harmonics)]
        freqs_to_remove = np.unique(freqs_to_remove)

        # Construct matrix A with on the columns sines and cosines of the
        # SSVEP fruencies and their harmonics
        A = np.tile(np.arange(self.nsamples, dtype=float), (2*len(freqs_to_remove), 1)).T
        A /= self.sample_rate
        A *= 2 * np.pi * freqs_to_remove.repeat(2)
        A[:, ::2] = np.sin(A[:, ::2])
        A[:, 1::2] = np.cos(A[:, 1::2])

        # Calculate inverse projection matrix P_A
        P_A =  A.dot( np.linalg.inv(A.T.dot(A)) ).dot(A.T)

        # Determine spatial filter that will remove all SSVEP signals
        self.SSVEPRemovalMatrix = (np.eye(self.nsamples) - P_A).T 

    def apply_(self, d):
        _, nsamples, ntrials = d.ndX.shape
        assert nsamples >= self.nsamples, 'Filter trained for a different window size.'

        filtered_ndX = np.zeros(d.ndX.shape)
        for trial in range(ntrials):
            Y = d.ndX[:, -self.nsamples:, trial]
            # TODO: normalize EEG signal power. All channels should have equal energy (variance)
            
            # Remove all SSVEP frequencies from the EEG
            tildeY = Y.dot(self.SSVEPRemovalMatrix)
            
            # Eigenvalue decomposition of remaining noise
            eigvals, eigvecs = np.linalg.eig(tildeY.dot(tildeY.T))
            eigvals[eigvals < 1e-10] = 0

            # Sort eigenvalues and eigenvectors from large to small
            sorted_ind = np.argsort(eigvals)[::-1]
            eigvals = eigvals[sorted_ind]
            eigvecs = eigvecs[:, sorted_ind]

            # Normalize eigenvectors by their eigenvalues
            eigvecs /= np.sqrt(eigvals)

            # Determine number of components to keep.
            eigvals_sum = np.sum(eigvals)
            for ncomponents_to_keep in range(len(eigvals)):
                if np.sum(eigvals[:ncomponents_to_keep]) / eigvals_sum > self.retain:
                    break

            # Set the components *not* to keep to 0
            # TODO: Ask why this step is omitted
            #eigvecs[:, ncomponents_to_keep:] = 0

            self.W = eigvecs / np.linalg.norm(eigvecs)
            self.W[np.isnan(self.W)] = 0 # rude hack

            # Apply the obtained spatial filter
            filtered_ndX[:,:,trial] = self.W.T.dot(d.ndX[:,:,trial])
            
        return DataSet(ndX=filtered_ndX, default=d)

class MNEC(BaseNode):
    '''
    SSVEP classifier based on Minimal Noise Energy Combination (MNEC) [1].

    [1] Friman, O., Volosyak, I., & Gräser, A. (2007). Multiple channel
    detection of steady-state visual evoked potentials for brain-computer
    interfaces. IEEE transactions on bio-medical engineering, 54(4), 742–50.
    '''
    def __init__(self, sample_rate, frequencies, nharmonics=2, retain=0.1, ar_order=20, weights=None, nsamples=None):
        BaseNode.__init__(self)
        self.sample_rate = sample_rate
        self.frequencies = frequencies
        self.nfrequencies = len(frequencies)
        self.retain = retain
        self.ar_order = ar_order
        self.harmonics = np.arange(nharmonics+1) + 1
        self.nharmonics = len(self.harmonics)
        self.nsamples = nsamples
        self.weights = weights

        self.noise_filter = SSVEPNoiseReduce(sample_rate, frequencies, nharmonics, retain, nsamples)

    def train_(self, d):
        self.noise_filter.train_(d)

        # Prepare some calculations in advance and store them in matrix X
        time = np.arange(self.noise_filter.nsamples) / float(self.sample_rate)
        X = np.tile( (2*np.pi*time)[:, np.newaxis, np.newaxis, np.newaxis],
                     [1, 2, self.nharmonics, self.nfrequencies] )
            
        for i,freq in enumerate(self.frequencies):
            X[:,:,:,i] = X[:,:,:,i] * freq

        for i,harm in enumerate(self.harmonics):
            X[:,:,i,:] = X[:,:,i,:] * harm
            
        X[:,0,:,:] = np.sin(X[:,0,:,:])
        X[:,1,:,:] = np.cos(X[:,1,:,:])

        self.X = X

    def apply_(self, d):
        S = self.noise_filter.apply_(d).ndX
        nchannels, nsamples, ntrials = S.shape

        result = []
        for trial in range(ntrials):
            P = np.zeros((self.nharmonics, nchannels, self.nfrequencies))
            for freq in range(self.nfrequencies):
                for ch in range(nchannels):
                    for harm in range(self.nharmonics):
                        P[harm,ch,freq] = np.linalg.norm(S[ch,:,trial].dot(self.X[...,harm,freq]))            
            P = P**2
            
            tildeS = S[...,trial].dot(self.noise_filter.SSVEPRemovalMatrix)
            
            Pxx = np.zeros((np.ceil((nsamples+1)/2.0), nchannels))
            nPxxRows = Pxx.shape[0]
            
            for ch in range(nchannels):
                p = spectrum.pyule(tildeS[ch,:], self.ar_order, NFFT=nsamples)
                p()
                Pxx[:,ch] = p.psd
                
            sigma = np.zeros((self.nharmonics, nchannels, self.nfrequencies))
            div = self.sample_rate / float(nsamples)
            
            for f, freq in enumerate(self.frequencies):
                for ch in range(nchannels):
                    for h, harm in enumerate(self.harmonics):
                        ind = round(freq * harm / div)
                        sigma[h,ch,f] = np.mean(Pxx[max(0,ind-1):min(ind+2, nPxxRows-1),ch])
                        
            SNRs = np.reshape(P / sigma, (self.nharmonics*nchannels, self.nfrequencies))
            nSNRs = SNRs.shape[0]
            
            if self.weights == None:
                self.weights = np.ones(nSNRs) / float(nSNRs)
            else:
                if len(self.weights) > nSNRs:
                    self.weights = self.weights[:nSNRs]
                elif len(self.weights) < nSNRs:
                     raise ValueError('inconsistent weight vector size')
                        
            scores = self.weights.dot(SNRs)
            result.append(scores)

        X = np.array(result).T
        feat_shape = X.shape[:-1]
        feat_lab = ['%d Hz' % f for f in self.frequencies] 

        return DataSet(X=X, feat_shape=feat_shape, feat_lab=feat_lab, feat_nd_lab=None, default=d)

class CanonCorr(BaseNode):
    '''
    SSVEP classifier based on Canonical Correlation Analysis (CCA) [1].

    [1] Frequency recognition based on canonical correlation analysis for
    SSVEP-based BCIs. Lin, Zhonglin / Zhang, Changshui / Wu, Wei / Gao,
    Xiaorong, IEEE transactions on bio-medical engineering, 53 (12 Pt 2),
    p.2610-2614, Dec 2006
    '''
    def __init__(self, sample_rate, frequencies, nharmonics=2, nsamples=None):
        '''
        parameters:
            sample_rate - sample rate of the signal
            frequencies - SSVEP frequencies that should be optimized for
            nharmonics  - number of harmonics of the SSVEP frequencies to optimize for
            nsamples    - number of samples in the EEG trials, when specified,
                          training of the node can be done without training data.
        '''
        BaseNode.__init__(self)
        self.sample_rate = sample_rate
        self.frequencies = frequencies
        self.nfrequencies = len(frequencies)
        self.harmonics = np.arange(nharmonics+1) + 1
        self.nharmonics = len(self.harmonics)
        self.nsamples = None

    def train_(self, d):
        if self.nsamples == None:
            assert d != None, 'Cannot determine window length.'
            self.nsamples = d.ndX.shape[1]

        # Construct matrices Y with on the columns sines and cosines of the
        # SSVEP frequencies and their harmonics
        Ys = np.tile(np.arange(self.nsamples, dtype=float)[:, np.newaxis], (self.nfrequencies, 1, 2*self.nharmonics))
        Ys /= self.sample_rate

        for i,freq in enumerate(self.frequencies):
            Ys[i,:,:] *= 2 * np.pi * freq * self.harmonics.repeat(2)

        Ys[:, :, ::2] = np.sin(Ys[:, :, ::2])
        Ys[:, :, 1::2] = np.cos(Ys[:, :, 1::2])

        # Perform Q-R decomposition on the frequencies
        QYs = []
        for freq in range(self.nfrequencies):
            Y = Ys[freq,:,:]
            Y = Y - np.tile(np.mean(Y, axis=0), (self.nsamples, 1))
            QY,_ = np.linalg.qr(Y)
            QYs.append(QY)

        self.QYs = QYs

    def apply_(self, d):
        nchannels, nsamples, ntrials = d.ndX.shape
        assert nsamples >= self.nsamples, 'Node trained for a different window size.'

        # Perform Q-R decomposition on the trials
        QXs = []
        for trial in range(ntrials):
            # The matrix X is our EEG signal, but we transpose it so
            # observations (=samples) are on the rows and variables (=channels)
            # are on the columns.
            X = d.ndX[:,:,trial].T

            # Center the variables (= remove the mean)
            X = X - np.tile(np.mean(X, axis=0), (nsamples, 1))
        
            QX,_ = np.linalg.qr(X)
            QXs.append(QX)

            
        # Calculate canonical correlations between trials and frequencies
        scores = np.zeros((self.nfrequencies, ntrials))
        for trial in range(ntrials):
            for freq in range(self.nfrequencies):
                # Compute canonical correlations through SVD
                _, D, _ = np.linalg.svd(QXs[trial].T.dot(self.QYs[freq]))
                
                # Note the first coefficient as the score
                scores[freq, trial] = D[0]

        feat_shape = scores.shape[:-1]
        feat_lab = ['%d Hz' % f for f in self.frequencies] 

        return DataSet(X=scores, feat_shape=feat_shape, feat_lab=feat_lab, feat_nd_lab=None, default=d)

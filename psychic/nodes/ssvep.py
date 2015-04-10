# coding=utf-8

import numpy as np
import scipy.linalg
from ..dataset import DataSet
from . import BaseNode
import itertools

def create_sin_cos_matrix(freqs, nharmonics, sample_rate, nsamples):
    '''
    Construct matrix labels with on the columns sines and cosines of the
    SSVEP frequencies and their harmonics.
    '''
    nfreqs = len(freqs)
    harmonics = np.arange(nharmonics+1) + 1

    Ys = np.tile(np.arange(nsamples, dtype=float)[:, np.newaxis],
                 (nfreqs, 1, 2*(nharmonics+1)))
    Ys /= sample_rate

    for i,freq in enumerate(freqs):
        Ys[i,:,:] *= 2 * np.pi * freq * harmonics.repeat(2)

    Ys[:, :, ::2] = np.sin(Ys[:, :, ::2])
    Ys[:, :, 1::2] = np.cos(Ys[:, :, 1::2])

    return Ys

class SLIC(BaseNode) :
    '''
    Node that applies the SLIC [1] algorithm to the data. The algorithm
    receives a number of frequencies as input. For each frequency, it splits
    the data in segments of 1/freq length and computes the correlation between
    each segment and the average. Signals with a high periodicity of the given
    frequency (such as SSVEP signals) will have a high correlation value,
    signals with a low periodicity or a periodicity with a different frequency
    will have a low correlation.

    Parameters
    ----------
    frequencies : list of floats
        The possible target frequencies.
    sample_rate : float (default: None)
        The sample rate of the data. When omitted, this is inferred during
        the training step. When specified, no training of the node is
        required.

    References
    ----------
    [1] Manyakov, Nikolay V, Nikolay Chumerin, Adrien Combaz, Arne Robben, and
    Marc M Van Hulle. 2010.  "Decoding SSVEP Responses Using Time Domain
    Classification." in International Conference on Neural Computation.
    Valentia, Spain.
    '''
    def __init__(self, frequencies, sample_rate=None):
        BaseNode.__init__(self)
        self.sample_rate = sample_rate
        self.frequencies = frequencies
        self.nfrequencies = len(frequencies)

    def train_(self, d):
        if d is None:
            assert self.sample_rate != None, 'Cannot determine sample rate.'
        else:
            if self.sample_rate is None:
                self.sample_rate = psychic.get_samplerate(d)

    def apply_(self, d):
        if d.ninstances == 0:
            return DataSet(default=d)

        mean_corrs = self.slic(d)

        data = np.max(mean_corrs, axis=1)
        feat_lab = ['%d Hz' % f for f in self.frequencies] 

        return DataSet(data=data, feat_lab=feat_lab, default=d)

    def slic(self, d):
        nchannels, nsamples, ntrials = d.data.shape

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
                periods = d.data[:,period_idxs,trial]

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
    classification using the method described in [1].

    This node performs some calculations during the training phase. However,
    when the ``sample_rate`` and ``nsamples`` parameters are supplied, this
    node does not need any training data (set it to ``None``). If they are not
    supplied, these are inferred from the training data. 

    Parameters
    ----------
    frequencies : list of floats
        SSVEP frequencies that should be optimized for.
    nharmonics : int (default: 2)
        Number of harmonics of the SSVEP frequencies to optimize for.
    retain : float between 0 and 1 (default: 0.1)
        Amount of noise variation to retain.
    sample_rate : float (default: None)
        The sample rate of the data. When omitted, this is inferred during
        the training step. When specified along with ``nsamples``, no training
        of the node is required.
    nsamples : int (default: None)
        Number of samples in each trial. When omitted, this is inferred during
        the training step. When specified along with ``sample_rate``, no
        training of the node is required.

    References
    ----------
    [1] Friman, O., Volosyak, ids., & Gräser, A. (2007). Multiple channel
    detection of steady-state visual evoked potentials for brain-computer
    interfaces.  IEEE transactions on bio-medical engineering, 54(4), 742–50.
    '''
    def __init__(self, frequencies, nharmonics=2, retain=0.1, sample_rate=None,
            nsamples=None):
        BaseNode.__init__(self)
        self.sample_rate = sample_rate
        self.frequencies = frequencies
        self.retain = retain
        self.harmonics = np.arange(nharmonics+1) + 1
        self.nharmonics = len(self.harmonics)
        self.nsamples = nsamples

    def reset(self):
        self.SSVEPRemovalMatrix = None

    def train_(self, d):
        if d is None:
            assert self.sample_rate != None, 'Cannot determine sample rate.'
            assert self.nsamples != None, 'Cannot determine window length.'
        else:
            if self.sample_rate is None:
                self.sample_rate = psychic.get_samplerate(d)
            if self.nsamples is None:
                self.nsamples = d.data.shape[1]

        # Construct a list of the SSVEP frequencies plus their harmonics
        freqs_to_remove = [f*h for f,h in itertools.product(self.frequencies, self.harmonics)]
        freqs_to_remove = np.unique(freqs_to_remove)

        # Construct matrix A with on the columns sines and cosines of the
        # SSVEP frequencies and their harmonics
        A = np.tile(np.arange(self.nsamples, dtype=float), (2*len(freqs_to_remove), 1)).T
        A /= self.sample_rate
        A *= 2 * np.pi * freqs_to_remove.repeat(2)
        A[:, ::2] = np.sin(A[:, ::2])
        A[:, 1::2] = np.cos(A[:, 1::2])

        # Calculate inverse projection matrix P_A
        P_A = A.dot( np.linalg.inv(A.T.dot(A)) ).dot(A.T)

        # Determine spatial filter that will remove all SSVEP signals
        self.SSVEPRemovalMatrix = (np.eye(self.nsamples) - P_A).T 

    def apply_(self, d):
        _, nsamples, ntrials = d.data.shape
        assert nsamples >= self.nsamples, 'Filter trained for a different window size.'

        filtered_ndX = np.zeros(d.data.shape)
        for trial in range(ntrials):
            labels = d.data[:, -self.nsamples:, trial]
            # TODO: normalize EEG signal power. All channels should have equal energy (variance)
            
            # Remove all SSVEP frequencies from the EEG
            tildeY = labels.dot(self.SSVEPRemovalMatrix)
            
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
            # eigvecs[:, ncomponents_to_keep:] = 0

            self.W = eigvecs / np.linalg.norm(eigvecs)
            self.W[np.isnan(self.W)] = 0 # rude hack

            # Apply the obtained spatial filter
            filtered_ndX[:,:,trial] = self.W.T.dot(d.data[:,:,trial])
            
        return DataSet(data=filtered_ndX, default=d)


try:
    import spectrum
    class MNEC(BaseNode):
        '''
        SSVEP classifier based on Minimal Noise Energy Combination (MNEC) [1].

        This node depends on the `Spectrum Analysis Tools
        <https://pypi.python.org/pypi/spectrum>`_ package.

        This node performs some calculations during the training phase.
        However, when the ``sample_rate`` and ``nsamples`` parameters are
        supplied, this node does not need any training data (set it to
        ``None``). If they are not supplied, these are inferred from the
        training data. 

        Parameters
        ----------
        frequencies : list of floats
            SSVEP frequencies that should be optimized for.
        nharmonics : int (default: 2)
            Number of harmonics of the SSVEP frequencies to optimize for.
        retain : float between 0 and 1 (default: 0.1)
            Amount of noise variation to retain.
        sample_rate : float (default: None)
            The sample rate of the data. When omitted, this is inferred during
            the training step. When specified along with ``nsamples``, no
            training of the node is required.
        nsamples : int (default: None)
            Number of samples in each trial. When omitted, this is inferred
            during the training step. When specified along with
            ``sample_rate``, no training of the node is required.

        References
        ----------
        [1] Friman, O., Volosyak, ids., & Gräser, A. (2007). Multiple channel
        detection of steady-state visual evoked potentials for brain-computer
        interfaces. IEEE transactions on bio-medical engineering, 54(4), 742–50.
        '''
        def __init__(self, frequencies, nharmonics=2, retain=0.1, ar_order=20,
                weights=None, sample_rate=None, nsamples=None):
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

            self.noise_filter = SSVEPNoiseReduce(frequencies, nharmonics, retain, sample_rate, nsamples)

        def reset(self):
            self.noise_filter.reset()
            self.data = None

        def train_(self, d):
            self.noise_filter.train_(d)

            # Prepare some calculations in advance and store them in matrix data
            time = np.arange(self.noise_filter.nsamples) / float(self.noise_filter.sample_rate)
            data = np.tile( (2*np.pi*time)[:, np.newaxis, np.newaxis, np.newaxis],
                         [1, 2, self.nharmonics, self.nfrequencies] )
                
            for i,freq in enumerate(self.frequencies):
                data[:,:,:,i] = data[:,:,:,i] * freq

            for i,harm in enumerate(self.harmonics):
                data[:,:,i,:] = data[:,:,i,:] * harm
                
            data[:,0,:,:] = np.sin(data[:,0,:,:])
            data[:,1,:,:] = np.cos(data[:,1,:,:])

            self.data = data

        def apply_(self, d):
            S = self.noise_filter.apply_(d).data[:, -self.noise_filter.nsamples:, :]
            nchannels, nsamples, ntrials = S.shape

            data = np.zeros((self.nfrequencies, d.ninstances))
            for trial in range(ntrials):
                P = np.zeros((self.nharmonics, nchannels, self.nfrequencies))
                for freq in range(self.nfrequencies):
                    for ch in range(nchannels):
                        for harm in range(self.nharmonics):
                            P[harm,ch,freq] = np.linalg.norm(S[ch,:,trial].dot(self.data[...,harm,freq]))            
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
                
                if self.weights is None:
                    self.weights = np.ones(nSNRs) / float(nSNRs)
                else:
                    if len(self.weights) > nSNRs:
                        self.weights = self.weights[:nSNRs]
                    elif len(self.weights) < nSNRs:
                         raise ValueError('inconsistent weight vector size')
                            
                data[:,trial] = self.weights.dot(SNRs)

            feat_lab = ['%d Hz' % f for f in self.frequencies] 

            return DataSet(data=data, feat_lab=feat_lab, default=d)
except:
    class MNEC(BaseNode):
        def __init__(self):
            raise RuntimeError("Cannot find required module: 'spectrum'.")

class CanonCorr(BaseNode):
    '''
    SSVEP classifier based on Canonical Correlation Analysis (CCA) [1].

    This node performs some calculations during the training phase.
    However, when the ``sample_rate`` and ``nsamples`` parameters are
    supplied, this node does not need any training data (set it to
    ``None``). If they are not supplied, these are inferred from the
    training data. 

    Parameters
    ----------
    frequencies : list of floats
        SSVEP frequencies that should be optimized for.
    nharmonics : int (default: 2)
        Number of harmonics of the SSVEP frequencies to optimize for.
    sample_rate : float (default: None)
        The sample rate of the data. When omitted, this is inferred during the
        training step. When specified along with ``nsamples``, no training of
        the node is required.
    nsamples : int (default: None)
        Number of samples in each trial. When omitted, this is inferred during
        the training step. When specified along with ``sample_rate``, no
        training of the node is required.

    References
    ----------
    [1] Frequency recognition based on canonical correlation analysis for
    SSVEP-based BCIs. Lin, Zhonglin / Zhang, Changshui / Wu, Wei / Gao,
    Xiaorong, IEEE transactions on bio-medical engineering, 53 (12 Pt 2),
    p.2610-2614, Dec 2006
    '''
    def __init__(self, frequencies, nharmonics=2, sample_rate=None, nsamples=None):
        BaseNode.__init__(self)
        self.sample_rate = sample_rate
        self.frequencies = frequencies
        self.nfrequencies = len(frequencies)
        self.nharmonics = nharmonics
        self.nsamples = nsamples

    def reset(self):
        self.QYs = None

    def train_(self, d):
        if d is None:
            assert self.sample_rate != None, 'Cannot determine sample rate.'
            assert self.nsamples != None, 'Cannot determine window length.'
        else:
            if self.sample_rate is None:
                self.sample_rate = psychic.get_samplerate(d)
            if self.nsamples is None:
                self.nsamples = d.data.shape[1]

        Ys = create_sin_cos_matrix(self.frequencies, self.nharmonics,
                self.sample_rate, self.nsamples)

        # Perform Q-R decomposition on the frequencies
        QYs = []
        for freq in range(self.nfrequencies):
            labels = Ys[freq,:,:]
            labels = labels - np.tile(np.mean(labels, axis=0), (self.nsamples, 1))
            QY,_ = np.linalg.qr(labels)
            QYs.append(QY)

        self.QYs = QYs

    def apply_(self, d):
        assert d.data.shape[1] >= self.nsamples, 'Node trained for a different window size.'
        nchannels, _, ntrials = d.data.shape

        # Perform Q-R decomposition on the trials
        QXs = []
        for trial in range(ntrials):
            # The matrix data is our EEG signal, but we transpose it so
            # observations (=samples) are on the rows and variables (=channels)
            # are on the columns.
            data = d.data[:,-self.nsamples:,trial].T

            # Center the variables (= remove the mean)
            data = data - np.tile(np.mean(data, axis=0), (self.nsamples, 1))
        
            QX,_ = np.linalg.qr(data)
            QXs.append(QX)

            
        # Calculate canonical correlations between trials and frequencies
        scores = np.zeros((self.nfrequencies, ntrials))
        for trial in range(ntrials):
            for freq in range(self.nfrequencies):
                # Compute canonical correlations through SVD
                _, D, _ = np.linalg.svd(QXs[trial].T.dot(self.QYs[freq]))
                
                # Note the first coefficient as the score
                scores[freq, trial] = D[0]

        feat_lab = ['%d Hz' % f for f in self.frequencies] 

        return DataSet(data=scores, feat_lab=feat_lab, default=d)

class MSI(BaseNode):
    '''
    SSVEP classifier based on Multivariate Synchronization Index (MSI) [1].

    This node performs some calculations during the training phase.
    However, when the ``sample_rate`` and ``nsamples`` parameters are
    supplied, this node does not need any training data (set it to
    ``None``). If they are not supplied, these are inferred from the
    training data. 

    Parameters
    ----------
    frequencies : list of floats
        SSVEP frequencies that should be optimized for.
    nharmonics : int (default: 2)
        Number of harmonics of the SSVEP frequencies to optimize for.
    sample_rate : float (default: None)
        The sample rate of the data. When omitted, this is inferred during the
        training step. When specified along with ``nsamples``, no training of
        the node is required.
    nsamples : int (default: None)
        Number of samples in each trial. When omitted, this is inferred during
        the training step. When specified along with ``sample_rate``, no
        training of the node is required.

    References
    ----------
    [1] Zhang, labels., Xu, P., Cheng, K., & Yao, D. (2013). Multivariate
    Synchronization Index for Frequency Recognition of SSVEP-based
    Brain-computer Interface. Journal of neuroscience methods, 1–9.
    doi:10.1016/j.jneumeth.2013.07.018
    '''
    def __init__(self, frequencies, nharmonics=2, sample_rate=None, nsamples=None):
        BaseNode.__init__(self)
        self.sample_rate = sample_rate
        self.frequencies = frequencies
        self.nfrequencies = len(frequencies)
        self.nharmonics = nharmonics
        self.nsamples = nsamples

    def reset(self):
        self.Ys = None
        self.C = None

    def train_(self, d):
        if d is None:
            assert self.sample_rate != None, 'Cannot determine sample rate.'
            assert self.nsamples != None, 'Cannot determine window length.'
        else:
            if self.sample_rate is None:
                self.sample_rate = psychic.get_samplerate(d)
            if self.nsamples is None:
                self.nsamples = d.data.shape[1]

        self.Ys = create_sin_cos_matrix(self.frequencies, self.nharmonics,
            self.sample_rate, self.nsamples)

        self.C22s = [labels.T.dot(labels) / float(self.nsamples) for labels in self.Ys]
        self.invC22s = [np.linalg.pinv(scipy.linalg.sqrtm(C22)) for C22 in self.C22s]

    def apply_(self, d):
        assert d.data.shape[1] >= self.nsamples, 'Node trained for a different window size.'
        nchannels, _, ntrials = d.data.shape

        scores = np.zeros((self.nfrequencies, ntrials))
        for trial in range(d.ninstances):
            data = d.data[:,:,trial]
            C11 = data.dot(data.T) / float(self.nsamples)
            invC11 = np.linalg.pinv(scipy.linalg.sqrtm(C11)) 

            for freq in range(self.nfrequencies):
                # Formula 4
                C12 = data.dot(self.Ys[freq]) / float(self.nsamples)
                C21 = C12.T

                # Formula 6
                dim = nchannels + 2 * (self.nharmonics + 1)
                R = np.eye(dim, dtype=np.complex)
                R[:nchannels, nchannels:] = invC11.dot(C12).dot(self.invC22s[freq])
                R[nchannels:, :nchannels] = self.invC22s[freq].dot(C21).dot(invC11)
        
                # Formula 7
                eigs = np.linalg.eigvals(R)
                eigs_norm = eigs / sum(eigs)

                # Prevent divide by 0 errors
                eigs_norm[eigs_norm < 1e-12] = 1e-12
                
                # Formula 8
                S = 1 + sum(eigs_norm * np.log(eigs_norm)) / np.log(dim)

                if np.isnan(S):
                    S = 0

                scores[freq, trial] = np.real(S)

        feat_lab = ['%d Hz' % f for f in self.frequencies] 

        return DataSet(data=scores, feat_lab=feat_lab, default=d)

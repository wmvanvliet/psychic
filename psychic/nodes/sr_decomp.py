import golem, numpy as np
from golem.nodes import BaseNode
from numpy.fft import fft, ifft

class SRDecomp(BaseNode):
    '''
    Temporally decompose single-channel data, such as electroencephalography
    (EEG) data, during reaction time tasks into a stimulus- and response-locked
    components using reaction times. Only keep stimulus-locked components
    '''

    def __init__(self, rt):
        BaseNode.__init__(self)
        self.rt = rt

    def train_(self, d):
        assert len(d.feat_shape) == 2, 'This node assumes the data to be split in trials'
        nchannels, nsamples, ntrials = d.ndX.shape; 
        assert (np.min(self.rt) >= 0 and np.max(self.rt) < nsamples), 'rt should be between 0 and the size of the first dimension of data'
        assert (len(self.rt) == ntrials), 'The length of rt should be the same as the size of the second dimension of data'

        # Calculate the FFT of the signal
        Y = fft(d.ndX, axis=1)
        self.Y_bar = np.mean(Y, axis=2) # Formula 4 in paper

        # Recalculate the rt as a phase shift in the FFT
        E = np.zeros((nsamples, ntrials), dtype=np.complex)
        for i in range(ntrials):
            E[:,i] = np.exp(-1j * 2 * np.pi * np.arange(nsamples) * self.rt[i] / complex(nsamples)) # Formula 5 in paper
        self.E_bar = np.mean(E, axis=1)
        
    def apply_(self, d):
        assert len(d.feat_shape) == 2, 'This node assumes the data to be split in trials'
        nchannels, nsamples, ntrials = d.ndX.shape; 
        assert (np.min(self.rt) >= 0 and np.max(self.rt) < nsamples), 'rt should be between 0 and the size of the first dimension of data'
        assert (len(self.rt) == ntrials), 'The length of rt should be the same as the size of the second dimension of data'

        # Calculate the FFT of the signal
        Y = fft(d.ndX, axis=1)

        # Recalculate the rt as a phase shift in the FFT
        E = np.zeros((nsamples, ntrials), dtype=np.complex)
        for i in range(ntrials):
            E[:,i] = np.exp(-1j * 2 * np.pi * np.arange(nsamples) * self.rt[i] / complex(nsamples)) # Formula 5 in paper

        # Preconstruct the denominator
        D = np.zeros((nsamples, ntrials), dtype=np.complex)
        for i in range(ntrials):
            D[:,i] = E[:,i] - self.E_bar
        D[0,:] = 1 # To prevent dividing by 0

        # Estimate the stimulus locked component
        sc = np.zeros(d.ndX.shape)
        for i in range(ntrials):
            for j in range(nchannels):
                sc[j,:,i] = np.real(ifft(
                    (E[:,i]*self.Y_bar[j,:] - self.E_bar*Y[j,:,i]) / D[:,i], # Formula 10 in paper
                    axis=0))

        return golem.DataSet(ndX=sc, default=d)

def sr_decomp(s, rt):
    '''
    Temporally decompose single-channel data, such as electroencephalography
    (EEG) data, during reaction time tasks into a stimulus- and response-locked
    components using reaction times.

    Usage: 
    >> (sc,rc) = sr_decomp(data,rt);

    Input:
     data - Single-channel data to decompose.
            Format (samples,channels,trials)
     rt   - Number of data points between stimulus and response onset.
            Format (trials,1)

    Output:
     sc  - Extracted stimulus-locked component.
            Format (samples,channels,trials)
            Number of data points before stimulus onset is 'stim' in 
            the input.
     rc  - Extracted response-locked component.
            Format (samples,channels,trials)
            Number of data points before response onset is 'stim' in 
            the input.

    The algorithm was taken from:
    Takeda, Y., Yamanaka, K., & Yamamoto, Y. (2008). Temporal decomposition of
    EEG during a simple reaction time task into stimulus- and response-locked
    components. NeuroImage, 39(2), 742-54. doi:10.1016/j.neuroimage.2007.09.003

    Original Matlab code was written by:
    Yusuke Takeda
    Educational Physiology Laboratory
    Graduate School of Education, The University of Tokyo
    7-3-1 Hongo, Bunkyo-ku, Tokyo 113-0033, Japan

    Python implementation by:
    Marijn van Vliet
    Department of Neurophysiology
    KU Leuven
    '''

    nchannels, nsamples, ntrials = s.shape; 
    assert (np.min(rt) >= 0 and np.max(rt) < nsamples), 'rt should be between 0 and the size of the first dimension of data'
    assert (len(rt) == ntrials), 'The length of rt should be the same as the size of the second dimension of data'

    # Calculate the FFT of the signal
    Y = fft(s, axis=1)
    Y_bar = np.mean(Y, axis=2) # Formula 4 in paper

    # Recalculate the rt as a phase shift in the FFT
    E = np.zeros((nsamples, ntrials), dtype=np.complex)
    for i in range(ntrials):
        E[:,i] = np.exp(-1j * 2 * np.pi * np.arange(nsamples) * rt[i] / complex(nsamples)) # Formula 5 in paper
    E_bar = np.mean(E, axis=1)

    # Preconstruct the denominator
    D = np.zeros((nsamples, ntrials), dtype=np.complex)
    for i in range(ntrials):
        D[:,i] = E[:,i] - E_bar
    D[0,:] = 1 # To prevent dividing by 0

    # Estimate the stimulus locked component
    sc = np.zeros(s.shape)
    for i in range(ntrials):
        for j in range(nchannels):
            sc[j,:,i] = np.real(ifft(
                (E[:,i]*Y_bar[j,:] - E_bar*Y[j,:,i]) / D[:,i], # Formula 10 in paper
                axis=0))

    # Estimate the response locked component
    rc = np.zeros(s.shape)
    for i in range(ntrials):
        for j in range(nchannels):
            rc[j,:,i] = np.real(ifft(
                (Y[j,:,i] - Y_bar[j,:]) / D[:,i], # Formula 11 in paper
                axis=0))

    return (sc, rc)

import numpy as np
import scipy as sci
from golem import DataSet
from golem.nodes import BaseNode
from matplotlib.mlab import PCA

class Slic(BaseNode) :
    """ Node that applies the SLIC [1] algorithm to the data. The algorithm
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
    """ 

    def __init__(self, freqs, samplerate, on_class=0, off_class=1):
        """ Create a new SLIC node.

        Required parameters:
        freqs: A list of frequencies to test for
        samplerate: The samplerate of the data. Needed, because the get_samplerate() utility
                    function of psychic only works before windowing of the data.
        """
        BaseNode.__init__(self)
        self.freqs = freqs
        self.samplerate = samplerate
        self.on_class = on_class
        self.off_class = off_class

    def train_(self, d):
        #if d.ninstances == 0:
        #    return

        #self.best_channels = []

        #mean_corrs = self.slic(d)

        #meanOn = np.mean(mean_corrs[d.ys[:,self.on_class],:,:], axis=0)
        #meanOff = np.mean(mean_corrs[d.ys[:,self.off_class],:,:], axis=0)
        #self.meanDifference = meanOn - meanOff

        #self.best_channels = np.argmax(self.meanDifference, axis=1)
        pass

    def apply_(self, d):
        if d.ninstances == 0:
            return DataSet(default=d)

        mean_corrs = self.slic(d)

        xs = []
        for freq_index, freq in enumerate(self.freqs):
            xs.append( np.max(mean_corrs, axis=2) )

        #xs = mean_corrs
        #return DataSet( np.vstack(xs), feat_shape=(len(self.freqs),mean_corrs.shape[2]), default=d )
        return DataSet( np.vstack(xs), feat_shape=(len(self.freqs),), default=d )

    def slic(self, d):
        num_samples, data_length, num_channels = d.nd_xs.shape
        num_freqs = len(self.freqs)

        if num_samples == 0:
            return DataSet(default=d)

        mean_corrs = np.zeros( (num_samples, num_freqs, num_channels) )
        for freq_index,freq in enumerate(self.freqs):
            # Determine the number and length of the periods
            period_length = self.samplerate/float(freq)
            num_periods = int(data_length/period_length)

            for sample in range(num_samples):
                # Construct the period indices
                offsets = np.round( np.arange(period_length, data_length, period_length) - period_length )
                length = np.arange( int(period_length) )

                x,y = np.meshgrid(offsets, length)
                period_idxs = (x+y).astype(np.int).T

                # Split the signal into periods
                #periods = segment.reshape([num_periods, period_length, num_channels])
                periods = d.nd_xs[sample,period_idxs,:]

                # Calculate the mean of the periods
                avg = np.mean(periods, axis=0)

                # Calculate the mean correlation between each period and the mean
                for channel in range(num_channels):
                    corrs = []
                    for period in range(0, periods.shape[0]):
                        corr = np.corrcoef(avg[:,channel].T, periods[period,:,channel])
                        corrs.append( 0.5*(corr + 1) )
                    mean_corrs[sample,freq_index,channel] = np.mean(corrs)

        return mean_corrs

class SSVEPNoiseReduce(BaseNode):
    def __init__(self, freq, samplerate, retain=0.1, ssvep_class=0):
        BaseNode.__init__(self)
        self.samplerate = samplerate
        self.freq = freq
        self.retain = retain
        self.ssvep_class = ssvep_class
        self.num_components = 5

    def train_(self, d):
        (num_windows, window_size, num_channels) = d.nd_xs.shape

        # Create array with two columns: [sin(2*PI*freq*sample), cos(2*Pi*freq*sample)]
        self.log.debug("Constructing Pa")

        # Calculate the projection matrix Pa to estimate useful signals
        A = np.atleast_2d( np.arange(window_size, dtype=np.double) ).repeat(2, axis=0).T
        A /= self.samplerate
        A *= 2*np.pi*self.freq
        A[:,0] = np.sin(A[:,0])
        A[:,1] = np.cos(A[:,1])

        # Store a spatial filter that will remove all SSVEP related signals
        self.SSVEPRemovalMatrix = np.eye(window_size) - A.dot( np.linalg.inv(A.T.dot(A)) ).dot(A.T)

    def apply_(self, d):
        (num_windows, window_size, num_channels) = d.nd_xs.shape

        filtered_xs = np.zeros([num_windows, window_size, self.num_components])

        for window in range(num_windows):
            # Calculate the PCA of the noise
            self.log.debug("Calculating PCA of noise")
            noise = self.SSVEPRemovalMatrix.dot(d.nd_xs[window,:,:])
            pca = PCA(noise)

            filtered_xs[window,:,:] = pca.Y[:,-self.num_components:]

            ## Determine the components to keep
            #var_explained = np.cumsum(pca.fracs[::-1]);
            #cutoff = np.where(var_explained <= self.retain)[0]

            #if cutoff.size == 0:
            #    self.log.warning("Could not determine a good noise model.")
            #    weights = np.eye(num_channels)
            #else:
            #    print 'pca.Wt[%d:,:]' % (-cutoff[-1]-1)
            #    print pca.Wt.shape
            #    weights = pca.Wt[-cutoff[-1]-1:] / sci.linalg.norm(pca.Wt[-cutoff[-1]-1:])

            #self.log.info('Noise reduced to: %g %%' % (100*var_explained[cutoff[-1]]))

            #print "Weights: "
            #print weights

            #print d.nd_xs[window,:,:].shape, weights.shape
            #filtered_xs[window,:,:] = np.atleast_2d( d.nd_xs[window,:,:].dot(weights.T) )

        return DataSet(filtered_xs.reshape(num_windows,-1), feat_shape=(window_size,self.num_components), default=d)

class MCD(BaseNode):
    def __init__(self, frequencies, samplerate, harmonics=2, nFFT=1000, AR_parameter=10):
        self.frequencies = np.asarray(frequencies)
        self.harmonics = np.arange(len(harmonics))
        self.samplerate = samplerate
        self.nFFT = nFFT
        self.AR_parameter = AR_parameter

    
    def train_(self, d):
        (window_size, num_channels, num_windows) = d.ndX.shape

        # Create array with two columns: [sin(2*PI*freq*sample), cos(2*Pi*freq*sample)]
        self.log.debug("Constructing Pa")

        # Calculate the projection matrix Pa to estimate useful signals
        A = np.atleast_2d( np.arange(window_size, dtype=np.double) ).repeat(2, axis=0).T
        A /= self.samplerate
        A *= 2*np.pi*self.frequencies.dot(self.harmonics)
        A[:,0] = np.sin(A[:,0])
        A[:,1] = np.cos(A[:,1])

        # Store a spatial filter that will remove all SSVEP related signals
        self.M = A.dot( np.linalg.inv(A.T.dot(A)) ).dot(A.T)
        self.SSVEPRemovalMatrix = np.eye(window_size) - self.M

    def apply_(self, data):
        return data

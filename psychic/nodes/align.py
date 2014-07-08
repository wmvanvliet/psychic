import golem
import numpy as np
from ..trials import erp
from spatialfilter import SpatialBlur

def channel_temporal_offsets(data, k=range(-20, 20)):
    '''
    Calculate temporal shift required for optimal cross channel covariances.

    parameters:
        data - (channels x samples) the signal
        k - maximum number of samples to temporally shift channels

    returns:
        Matrix (channels x channels) containing for each channel the temporal
        shift compared to all other channels.
    '''
    nchannels, nsamples = data.shape

    T = np.zeros((nchannels, nchannels))
    for m in range(nchannels):
        for n in range(m+1, nchannels):
            rs = []
            for t in k:
                if t > 0:                 
                    rs.append( np.cov(data[m,t:], data[n,:-t])[0,1] )
                elif t == 0:
                    rs.append( np.cov(data[m,:], data[n,:])[0,1] )
                else:
                    rs.append( np.cov(data[m,:t], data[n,-t:])[0,1] )
    
            if np.abs(np.max(rs)) < 0.5:
                T[m,n] = 0
            else:
                t = np.argmax(np.abs(rs))
                if t == 0 or t == len(rs)-1:
                    T[m,n] = 0
                else:
                    T[m,n] = k[t]
                
            T[n,m] = -T[m,n]
            
    return T

class AlignedSpatialBlur(SpatialBlur):
    '''
    Same as a SpatialBlur filter, but first temporally aligns the
    channels in order to take propagation delay into account.

    [1] Ke Yu, Kaiquan Shen, Shiyun Shao, Wu Chun Ng, Kenneth Kwok, Xiaoping
    Li. A spatio-temporal filtering approach to denoising of single-trial ERP
    in rapid image triage. Journal of Neuroscience Methods, 204:288--295, 2012.

    parameters:
        sigma - standard deviation of the gaussian used in the spatial blur
        k     - maximum number of samples to temporally shift channels
    '''

    def __init__(self, sigma, k, ftype=1):
        SpatialBlur.__init__(self, sigma, ftype)
        self.k = range(-k, k)

    def train_(self, d):
        if self.ftype == 0:
            self.T = channel_temporal_offsets(d.data, self.k)
        elif self.ftype == 1:
            data = np.mean(erp(d).data, axis=2)
            self.T = channel_temporal_offsets(data, self.k)
        else:
            raise ValueError('Operation not supported on covariance data')

        SpatialBlur.train_(self, d)

    def apply_(self, d):
        t_min = np.min(self.T)
        t_max = np.max(self.T)

        nchannels, nsamples, ntrials = d.data.shape
        nsamples -= int(t_max - t_min)

        data = np.zeros((nchannels, nsamples, ntrials))
        for m in range(nchannels):
            for n in range(nchannels):
                t = self.T[m,n] - t_min
                data[m, ...] += self.W[m,n] * d.data[n,t:t+nsamples,...]
                
        feat_lab = list(d.feat_lab)
        feat_lab[1] = [float(t) + t_min for t in feat_lab[1][:nsamples]]
            
        return golem.DataSet(data=data, feat_lab=feat_lab, default=d)

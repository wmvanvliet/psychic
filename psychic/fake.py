from golem import DataSet
from scipy.stats import norm
import positions
import numpy as np

def sine(freq, nchannels, duration, sample_rate):
    '''
    Generate a fake dataset containing sine waves.

    Example:

    >>> import psychic
    >>> print psychic.fake.sine(10, 4, 1.5, 100)
    DataSet with 150 instances, 4 features [4], 1 classes: [150], extras: []

    Parameters
    ----------
    freq : float or list of floats
        The frequency of the sine waves in Hz. Can be a list supplying a
        different frequency for each channel.
    nchannels : int
        Desired number of channels.
    duration : float
        Desired duration (in seconds) of the data.
    sample_rate : float
        Desired sample rate of the data 

    Returns
    -------
    d : :class:`golem.DataSet`
        The generated data.

    '''
    time = np.arange(duration * sample_rate) / float(sample_rate)
    nsamples = len(time)

    try:
        iter(freq)
    except TypeError:
        freq = [freq for _ in range(nchannels)]

    X = np.array([np.sin(freq[x] * 2 * np.pi * time) for x in range(nchannels)])
    Y = np.zeros((1, nsamples))
    I = time
    feat_lab = ['CH %02d' % (ch+1) for ch in range(nchannels)]

    return DataSet(X=X, Y=Y, I=I, feat_lab=feat_lab)

def gaussian(nchannels, duration, sample_rate):
    '''
    Generate a fake dataset containing gaussian noise.

    Example:

    >>> import psychic
    >>> print psychic.fake.gaussian(4, 1.5, 100)
    DataSet with 150 instances, 4 features [4], 1 classes: [150], extras: []

    Parameters
    ----------
    nchannels : int
        Desired number of channels.
    duration : float
        Desired duration (in seconds) of the data.
    sample_rate : float
        Desired sample rate of the data 

    Returns
    -------
    d : :class:`golem.DataSet`
        The generated data.

    '''
    time = np.arange(duration * sample_rate) / float(sample_rate)
    nsamples = len(time)

    X = np.random.randn(nchannels, nsamples)
    Y = np.zeros((1, nsamples))
    I = time
    feat_lab = ['CH %02d' % (ch+1) for ch in range(nchannels)]

    return DataSet(X=X, Y=Y, I=I, feat_lab=feat_lab)

def generate_erp(time, channels, time_loc, time_scale, space_loc, space_scale, amp_scale):
    '''
    Generate an ERP
    '''
    X = np.empty((len(channels), len(time)))
    
    time_pdf = norm(loc=time_loc, scale=time_scale).pdf
    space_x_pdf = norm(loc=space_loc[0], scale=space_scale).pdf
    space_y_pdf = norm(loc=space_loc[1], scale=space_scale).pdf

    locs = np.array([positions.project_scalp(*positions.POS_10_5[lab]) for lab in channels])

    X = (space_x_pdf(locs[:,0]) * space_y_pdf(locs[:,1]))[:, np.newaxis].dot(time_pdf(time)[np.newaxis,:])
    X /= np.max(X) * amp_scale
    return X

from .dataset import DataSet
from .helpers import to_one_of_n
from scipy.stats import norm
from . import positions
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
    d : :class:`psychic.DataSet`
        The generated data.

    '''
    time = np.arange(duration * sample_rate) / float(sample_rate)
    nsamples = len(time)

    try:
        iter(freq)
    except TypeError:
        freq = [freq for _ in range(nchannels)]

    data = np.array([np.sin(freq[x] * 2 * np.pi * time) for x in range(nchannels)])
    ids = time
    feat_lab = ['CH %02d' % (ch+1) for ch in range(nchannels)]

    return DataSet(data=data, ids=ids, feat_lab=feat_lab)

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
    d : :class:`psychic.DataSet`
        The generated data.

    '''
    time = np.arange(duration * sample_rate) / float(sample_rate)
    nsamples = len(time)

    data = np.random.randn(nchannels, nsamples)
    ids = time
    feat_lab = ['CH %02d' % (ch+1) for ch in range(nchannels)]

    return DataSet(data=data, ids=ids, feat_lab=feat_lab)

def generate_erp(time, channels, time_loc, time_scale, space_loc, space_scale, amp_scale):
    '''
    Generate an ERP
    '''
    data = np.empty((len(channels), len(time)))
    
    time_pdf = norm(loc=time_loc, scale=time_scale).pdf
    space_x_pdf = norm(loc=space_loc[0], scale=space_scale).pdf
    space_y_pdf = norm(loc=space_loc[1], scale=space_scale).pdf

    locs = np.array([positions.project_scalp(*positions.POS_10_5[lab]) for lab in channels])

    data = (space_x_pdf(locs[:,0]) * space_y_pdf(locs[:,1]))[:, np.newaxis].dot(time_pdf(time)[np.newaxis,:])
    data /= np.max(data) * amp_scale
    return data

def gaussian_dataset(ninstances=[50, 50]):
    '''
    Simple Gaussian dataset with a variable number of instances for up to 3
    classes.
    '''
    mus = [\
        [0, 0], 
        [2, 1],
        [5, 6]]
    sigmas = [\
        [[1, 2], [2, 5]],
        [[1, 2], [2, 5]],
        [[1, -1], [-1, 2]]]

    assert len(ninstances) <= 3

    data, labels = [], []
    for (ci, n) in enumerate(ninstances):
        data.append(np.random.multivariate_normal(mus[ci], sigmas[ci], n).T)
        labels.extend(np.ones(n, np.int) * ci)

    return DataSet(np.hstack(data), to_one_of_n(labels))

def wieland_spirals():
    '''
    Famous non-linear binary 2D problem with intertwined spirals.
    '''
    i = np.arange(97)
    theta = np.pi * i / 16.
    r = 6.5 * (104 - i) / 104.
    data = np.array([r * np.cos(theta), r * np.sin(theta)])
    data = np.hstack([data, -data])
    labels = to_one_of_n(np.hstack([np.zeros(i.size), np.ones(i.size)]))
    return DataSet(data, labels)

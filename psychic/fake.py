from golem import DataSet
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

    return DataSet(X=X, Y=Y, I=I)

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

    X = np.random.randn((nchannels, nsamples))
    Y = np.zeros((1, nsamples))
    I = time

    return DataSet(X=X, Y=Y, I=I)

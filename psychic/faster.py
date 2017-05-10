import numpy as np
import psychic

from numpy.polynomial.legendre import legval
from scipy import linalg
from scipy.signal import welch, lfilter
from scipy.stats import kurtosis, zscore
from psychic.nodes.eeg_montage import _ch_idx
from psychic.nodes import BaseNode


def find_outliers(X, threshold=3.0, max_iter=2):
    """Find outliers based on iterated Z-scoring

    This procedure compares the absolute z-score against the threshold.
    After excluding local outliers, the comparison is repeated until no
    local outlier is present any more.

    Parameters
    ----------
    X : np.ndarray of float, shape (n_elemenets,)
        The scores for which to find outliers.
    threshold : float
        The value above which a feature is classified as outlier.
    max_iter : int
        The maximum number of iterations.

    Returns
    -------
    bad_idx : np.ndarray of int, shape (n_features)
        The outlier indices.
    """
    my_mask = np.zeros(len(X), dtype=np.bool)
    for _ in range(max_iter):
        X = np.ma.masked_array(X, my_mask)
        this_z = np.abs(zscore(X))
        local_bad = this_z > threshold
        my_mask = np.max([my_mask, local_bad], 0)
        if not np.any(local_bad):
            break

    bad_idx = np.where(my_mask)[0]
    return bad_idx


def _hurst(x):
    """Estimate Hurst exponent on a timeseries.

    The estimation is based on the second order discrete derivative.

    Parameters
    ----------
    x : 1D numpy array
        The timeseries to estimate the Hurst exponent for.

    Returns
    -------
    h : float
        The estimation of the Hurst exponent for the given timeseries.
    """
    y = np.cumsum(np.diff(x, axis=1), axis=1)

    b1 = [1, -2, 1]
    b2 = [1,  0, -2, 0, 1]

    # second order derivative
    y1 = lfilter(b1, 1, y, axis=1)
    y1 = y1[:, len(b1) - 1:-1]  # first values contain filter artifacts

    # wider second order derivative
    y2 = lfilter(b2, 1, y, axis=1)
    y2 = y2[:, len(b2) - 1:-1]  # first values contain filter artifacts

    s1 = np.mean(y1 ** 2, axis=1)
    s2 = np.mean(y2 ** 2, axis=1)

    return 0.5 * np.log2(s2 / s1)


def _efficient_welch(data, sfreq):
    """Calls scipy.signal.welch with parameters optimized for greatest speed
    at the expense of precision. The window is set to ~10 seconds and windows
    are non-overlapping.

    Parameters
    ----------
    data : N-D numpy array
        The timeseries to estimate signal power for. The last dimension
        is presumed to be time.
    sfreq : float
        The sample rate of the timeseries.

    Returns
    -------
    fs : 1D numpy array
        The frequencies for which the power spectra was calculated.
    ps : ND numpy array
        The power spectra for each timeseries.
    """
    nperseg = min(data.shape[1],
                  2 ** int(np.log2(10 * sfreq) + 1))  # next power of 2

    return welch(data, sfreq, nperseg=nperseg, noverlap=0, axis=1)


def _freqs_power(data, sfreq, freqs):
    """Estimate signal power at specific frequencies.

    Parameters
    ----------
    data : N-D numpy array
        The timeseries to estimate signal power for. The last dimension
        is presumed to be time.
    sfreq : float
        The sample rate of the timeseries.
    freqs : list of float
        The frequencies to estimate signal power for.

    Returns
    -------
    ps : list of float
        For each requested frequency, the estimated signal power.
    """
    fs, ps = _efficient_welch(data, sfreq)
    return np.sum([ps[..., np.searchsorted(fs, f)] for f in freqs], axis=0)


def _power_gradient(data, sfreq):
    """Estimate the gradient of the power spectrum at upper frequencies.

    Parameters
    ----------
    data : N-D numpy array
        The timeseries to estimate signal power for. The last dimension
        is presumed to be time.
    sfreq : float
        The sample rate of the timeseries.

    Returns
    -------
    grad : N-D numpy array
        The gradients of each timeseries.
    """
    fs, ps = _efficient_welch(data, sfreq)

    # Limit power spectrum to upper frequencies
    ps = ps[:, np.searchsorted(fs, 25):np.searchsorted(fs, 45)]

    # Compute mean gradients
    return np.mean(np.diff(ps), axis=1)


def _deviation(data):
    """Computes the deviation from mean for each channel in a set of epochs.

    This is not implemented as a lambda function, because the channel means
    should be cached during the computation.

    Parameters
    ----------
    data : 3D numpy array
        The epochs (#epochs x #channels x #samples).

    Returns
    -------
    dev : 2D numpy array
        For each epoch, the mean deviation of the channels.
    """
    ch_mean = np.mean(data, axis=1)
    return ch_mean - np.mean(ch_mean, axis=1)[:, np.newaxis]


def bad_channels(data, channels=None, thres=3, use_metrics=None):
    """Implements the first step of the FASTER algorithm.
    
    This function attempts to automatically mark bad EEG channels by performing
    outlier detection. It operated on epoched data, to make sure only relevant
    data is analyzed.

    Parameters
    ----------
    data : DataSet
        The continuous data or epochs for which bad channels need to be marked
    channels : list of (int | str) | None
        List of channels to operate on. Defaults to all channels. Channels
        can be specified by integer index or by string name.
    thres : float
        The threshold value, in standard deviations, to apply. A channel
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'variance', 'correlation', 'hurst', 'kurtosis', 'line_noise'
        Defaults to all of them.

    Returns
    -------
    bads : list of str
        The names of the bad EEG channels.
    """
    samplerate = psychic.get_samplerate(data)

    if channels is None:
        channels = list(range(data.data.shape[0]))
    else:
        channels = list(_ch_idx(channels, data.feat_lab[0]))

    metrics = {
        'variance':    lambda x: np.var(x, axis=1),
        'correlation': lambda x: np.mean(
                           np.ma.masked_array(
                               np.corrcoef(x),
                               np.identity(len(x), dtype=bool)
                           ),
                           axis=0),
        'hurst':       lambda x: _hurst(x),
        'kurtosis':    lambda x: kurtosis(x, axis=1),
        'line_noise':  lambda x: _freqs_power(x, samplerate, [50, 60]),
    }

    if use_metrics is None:
        use_metrics = list(metrics.keys())

    # Concatenate epochs in time
    if data.data.ndim == 3:
        data = psychic.concatenate_trials(data.ix[channels, :, :])
    elif data.data.ndim == 2:
        data = data.ix[channels, :]
    else:
        raise ValueError('Expected 2D or 3D data.')

    # Find bad channels
    bads = []
    for m in use_metrics:
        s = metrics[m](data.data)
        b = [data.feat_lab[0][i] for i in find_outliers(s, thres)]
        #logger.info('Bad by %s:\n\t%s' % (m, b))
        print('Bad by %s:\n\t%s' % (m, b))
        bads.append(b)

    return np.unique(np.concatenate(bads)).tolist()


def bad_epochs(epochs, channels=None, thres=3, use_metrics=None):
    """Implements the second step of the FASTER algorithm.
    
    This function attempts to automatically mark bad epochs by performing
    outlier detection.

    Parameters
    ----------
    epochs : DataSet
        The epochs to analyze.
    channels : list of (int | str) | None
        Channels to operate on. Defaults to all channels.
    thres : float
        The threshold value, in standard deviations, to apply. An epoch
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'amplitude', 'variance', 'deviation'
        Defaults to all of them.

    Returns
    -------
    bads : list of int
        The indices of the bad epochs.
    """
    if channels is None:
        channels = list(range(epochs.data.shape[0]))
    else:
        channels = list(_ch_idx(channels, epochs.feat_lab[0]))

    metrics = {
        'amplitude': lambda x: np.mean(np.ptp(x, axis=1), axis=0),
        'deviation': lambda x: np.mean(_deviation(x), axis=0),
        'variance':  lambda x: np.mean(np.var(x, axis=1), axis=0),
    }

    if use_metrics is None:
        use_metrics = list(metrics.keys())

    epochs = epochs.ix[channels, :, :]

    bads = []
    for m in use_metrics:
        s = metrics[m](epochs.data)
        b = find_outliers(s, thres)
        logger.info('Bad by %s:\n\t%s' % (m, b))
        bads.append(b)

    return np.unique(np.concatenate(bads)).tolist()


def bad_channels_in_epochs(epochs, channels=None, thres=3, use_metrics=None):
    """Implements the fourth step of the FASTER algorithm.
    
    This function attempts to automatically mark bad channels in each epochs by
    performing outlier detection.

    Parameters
    ----------
    epochs : DataSet
        The epochs to analyze.
    channels : list of int | None
        Channels to operate on. Defaults to all channels.
    thres : float
        The threshold value, in standard deviations, to apply. An epoch
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'amplitude', 'variance', 'deviation', 'median_gradient'
        Defaults to all of them.

    Returns
    -------
    bads : list of lists of int
        For each epoch, the indices of the bad channels.
    """
    samplerate = psychic.get_samplerate(epochs)

    if channels is None:
        channels = list(range(epochs.epochs.shape[0]))
    else:
        channels = list(_ch_idx(channels, epochs.feat_lab[0]))


    metrics = {
        'amplitude':       lambda x: np.ptp(x, axis=1),
        'deviation':       lambda x: _deviation(x),
        'variance':        lambda x: np.var(x, axis=1),
        'median_gradient': lambda x: np.median(np.abs(np.diff(x, axis=1)), axis=1),
        'line_noise':      lambda x: _freqs_power(x, samplerate, [50, 60]),
    }

    if use_metrics is None:
        use_metrics = list(metrics.keys())
    
    epochs = epochs.ix[channels, :, :]

    bads = [[] for i in range(len(epochs))]
    for m in use_metrics:
        s_epochs = metrics[m](epochs.data)
        for i, s in enumerate(s_epochs.T):
            b = [epochs.feat_lab[0][j] for j in find_outliers(s, thres)]
            if len(b) > 0:
                logger.info('Epoch %d, Bad by %s:\n\t%s' % (i, m, b))
            bads[i].append(b)

    for i, b in enumerate(bads):
        if len(b) > 0:
            bads[i] = np.unique(np.concatenate(b)).tolist()

    return bads


def _calc_g(cosang, stiffness=4, num_lterms=50):
    """Calculate spherical spline g function between points on a sphere.

    Parameters
    ----------
    cosang : array-like of float, shape(n_channels, n_channels)
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffness : float
        stiffness of the spline.
    num_lterms : int
        number of Legendre terms to evaluate.

    Returns
    -------
    G : np.ndrarray of float, shape(n_channels, n_channels)
        The G matrix.
    """
    factors = [(2 * n + 1) / (n ** stiffness * (n + 1) ** stiffness *
                              4 * np.pi) for n in range(1, num_lterms + 1)]
    return legval(cosang, [0] + factors)


def _calc_h(cosang, stiffness=4, num_lterms=50):
    """Calculate spherical spline h function between points on a sphere.

    Parameters
    ----------
    cosang : array-like of float, shape(n_channels, n_channels)
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffness : float
        stiffness of the spline. Also referred to as `m`.
    num_lterms : int
        number of Legendre terms to evaluate.
    H : np.ndrarray of float, shape(n_channels, n_channels)
        The H matrix.
    """
    factors = [(2 * n + 1) /
               (n ** (stiffness - 1) * (n + 1) ** (stiffness - 1) * 4 * np.pi)
               for n in range(1, num_lterms + 1)]
    return legval(cosang, [0] + factors)


def _normalize_vectors(rr):
    size = np.sqrt(np.sum(rr * rr, axis=1))
    size[size == 0] = 1.0  # avoid divide-by-zero
    rr /= size[:, np.newaxis]  # operate in-place


def _make_interpolation_matrix(pos_from, pos_to, alpha=1e-5):
    """Compute interpolation matrix based on spherical splines

    Implementation based on [1]

    Parameters
    ----------
    pos_from : np.ndarray of float, shape(n_good_sensors, 3)
        The positions to interpoloate from.
    pos_to : np.ndarray of float, shape(n_bad_sensors, 3)
        The positions to interpoloate.
    alpha : float
        Regularization parameter. Defaults to 1e-5.

    Returns
    -------
    interpolation : np.ndarray of float, shape(len(pos_from), len(pos_to))
        The interpolation matrix that maps good signals to the location
        of bad signals.

    References
    ----------
    [1] Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989).
        Spherical splines for scalp potential and current density mapping.
        Electroencephalography Clinical Neurophysiology, Feb; 72(2):184-7.
    """

    pos_from = pos_from.copy()
    pos_to = pos_to.copy()

    # normalize sensor positions to sphere
    _normalize_vectors(pos_from)
    _normalize_vectors(pos_to)

    # cosine angles between source positions
    cosang_from = pos_from.dot(pos_from.T)
    cosang_to_from = pos_to.dot(pos_from.T)
    G_from = _calc_g(cosang_from)
    G_to_from, H_to_from = (f(cosang_to_from) for f in (_calc_g, _calc_h))

    if alpha is not None:
        G_from.flat[::len(G_from) + 1] += alpha

    C_inv = linalg.pinv(G_from)
    interpolation = G_to_from.dot(C_inv)
    return interpolation


def interpolate_channels(data, channels):
    """Interpolate channels from surrounding channels.

    Parameters
    ----------
    data : DataSet
        The data to interpolate. Either continuous or as epochs.
    channels : list of (int | str)
        The channels to interpolate.
    """
    to_idx = list(_ch_idx(channels, data.feat_lab[0]))
    to_names = [data.feat_lab[0][ch] for ch in to_idx]
    from_idx = [i for i, ch in enumerate(data.feat_lab[0])
                if (i not in to_idx and ch in psychic.positions.POS_10_5)]
    from_names = [data.feat_lab[0][ch] for ch in from_idx]

    to_pos = np.array([psychic.positions.POS_10_5[ch] for ch in to_names])
    from_pos = np.array([psychic.positions.POS_10_5[ch]
                         for ch in from_names
                         if ch in psychic.positions.POS_10_5])

    interpolation = _make_interpolation_matrix(from_pos, to_pos)

    print('Interpolating %d sensors' % len(to_pos))

    data_ = data.data.copy()
    if data.data.ndim == 2:
        data_[to_idx, :] = interpolation.dot(data_[from_idx, :])
    elif data.data.ndim == 3:
        for i in range(data.data.shape[2]):
            data_[to_idx, :, i] = interpolation.dot(data_[from_idx, :, i])
    else:
        raise ValueError('Only 2D or 3D data is supported.')

    data = psychic.DataSet(data=data_, default=data)
    return data, interpolation

class InterpolateChannels(BaseNode):
    """Interpolate channels from surrounding channels.

    This implementation was adapted by the original one by Denis Engemann for
    the MNE-Python toolbox (https://github.com/mne-tools/mne-python/blob/master/mne/channels/interpolation.py).
    The original spline interpolation technique comes from:

    Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989). Spherical
    splines for scalp potential and current density mapping.
    Electroencephalography Clinical Neurophysiology, Feb; 72(2):184-7

    Parameters
    ----------
    channels : list of (int | str)
        The channels to interpolate. Channels can be specified by integer index
        or by string name.
    """
    def __init__(self, channels):
        BaseNode.__init__(self)
        self.channels = channels

    def apply_(self, d):
        if len(self.channels) == 0:
            return d
        else:
            return interpolate_channels(d, self.channels)[0]

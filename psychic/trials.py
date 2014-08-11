import numpy as np
import scipy
import logging
import helpers
import markers
import utils
from dataset import DataSet, concatenate
from matplotlib.mlab import specgram

def baseline(d, baseline_period=None):
    '''
    For each channel, calculate and remove the baseline. The baseline is the
    mean signal calculated over a certain time period.

    Parameters
    ----------
    baseline_period : tuple (int, int) (default: full range of samples)
        The start (inclusive) and end (exclusive) indices of the period to
        calculate the baseline over. Values are given in samples. 

    Returns
    -------
    d : :class:`psychic.DataSet`
        The baselined trials.
    '''
    if baseline_period:
        assert  len(baseline_period) == 2, \
          'Specify a begin and end point for the baseline period (in samples)'
    else:
        baseline_period = (0, d.ninstances)

    assert d.data.ndim <= 3

    if d.data.ndim == 2:
        num_samples = data.ninstances
        d = d.data - np.tile( np.mean(d.data[:,baseline_period[0]:baseline_period[1]], axis=1).T, (num_samples, 1) ).T

    else:
        num_samples = d.data.shape[1]
        data = np.zeros(d.data.shape, dtype=d.data.dtype)

        for i in range(d.ninstances):
            data[:,:,i] = d.data[:,:,i] - np.tile( np.mean(d.data[:,baseline_period[0]:baseline_period[1],i], axis=1).T, (num_samples, 1) ).T
        data = data.reshape(d.data.shape)

    return DataSet(data, default=d)

def erp(d, classes=None, enforce_equal_n=True):
    '''
    For each class, calculate the Event Related Potential by averaging the 
    corresponding trials. Note: no baselining is performed, see
    :func:`psychic.baseline`.

    Parameters
    ----------
    data : :class:`psychic.DataSet`
        The trials
    classes: list (optional)
        When specified, the ERP is only calculated for the classes with the
        given indices.
    enforce_equal_n : bool (default=True)
        When set, each ERP is calculated by averaging the same number of
        trials. For example, if class1 has m and class2 has n trials and m > n.
        The ERP for class1 will be calculated by taking n random trials from
        class1.
    
    Returns
    -------
    d : :class:`psychic.DataSet`
        A DataSet containing for each class the ERP. 

        - ``d.data``: [channels x samples x classes]
        - ``d.labels``: The class labels. Each class has one instance (one ERP).
    '''
    assert d.data.ndim > 2

    if classes == None or len(classes) == 0:
        # Take all classes with >0 instances
        classes = [cl for cl in range(d.nclasses)
                   if d.ninstances_per_class[cl] > 0]
    assert  len(classes) > 0, \
            'No valid classes specified and no classes found with >0 instances'

    num_trials = np.min( np.array(d.ninstances_per_class)[classes] )
    assert num_trials > 0, 'For one or more classes there are no instances!'

    # Calculate ERP
    erp = np.zeros(d.data.shape[:-1] + (len(classes),))
    for i,cl in enumerate(classes):
        trials = d.get_class(cl).data

        if enforce_equal_n:
            # Enforce an equal number of trials for all classes. Picking them
            # at random.  Otherwise the ERPs will be skewed, simply because a
            # different number of trials are averaged.
            idx = range(trials.shape[-1])[:num_trials]
            np.random.shuffle(idx)
            erp[...,i] = np.mean(trials[...,idx], axis=trials.ndim-1)
        else:
            erp[...,i] = np.mean(trials, axis=trials.ndim-1)

    labels = helpers.to_one_of_n(classes).astype(np.bool)
    ids = np.atleast_2d(classes)
    cl_lab = [lab for i,lab in enumerate(d.cl_lab) if i in classes]

    return DataSet(data=erp, labels=labels, ids=ids, cl_lab=cl_lab,
                   default=d)

def ttest(d, classes=[0, 1], shuffle=True):
    '''
    Calculate ttests between two classes for each channel and each sample. If
    one class has more trials than the other, random trials will be taken to
    ensure an equal number of trials.

    Parameters
    ----------
    d : :class:psychic.DataSet:
        The trials.
    classes : list (default=[0, 1])
        The indices of the classes to compare.
    shuffle : bool (default=False)
        When set, trials will be shuffled prior to comparison.

    Returns
    -------
    t-values: [channels x samples] array
        The t-values 

    p-values: [channels x samples] array
        The p-values 
    '''
    assert d.nd_xs.ndim == 3
    assert d.nclasses >= 2, ('Data must contain at least two classes ',
                             'otherwise there is nothing to compare.')

    num_trials = np.min( np.array(d.ninstances_per_class)[classes] )

    c1 = d.data[..., d.labels[classes[0],:].astype(np.bool)]
    if shuffle:
        np.random.shuffle(c1)
    c1 = c1[..., :num_trials]

    c2 = d.data[..., d.labels[classes[1],:].astype(np.bool)]
    if shuffle:
        np.random.shuffle(c2)
    c2 = c2[..., :num_trials]

    return scipy.stats.ttest_ind(c1, c2, axis=2)

def random_groups(d, group_size, groups_per_class=None, mean=False):
    '''
    For each class, form groups of random trials of the given size.

    Parameters
    ----------
    d : :class:`DataSet`
        The trials.
    group_size : int
        Size of the groups to make.
    groups_per_class : int (default=ninstances_per_class / group_size)
        Number of groups to make per class.
    mean : bool (default=False)
        Return group means (i.e. ERPs) instead of groups

    Returns
    -------
    d : :class:`DataSet`
        The grouped data. ``d.data`` is [channels x samples x trials x groups]
        or if mean==True, ``d.data`` is [channels x samples x groups]
    '''

    d_trials = []
    idxs = []

    if groups_per_class == None:
        groups_per_class = np.min(d.ninstances_per_class) / group_size

    for cl in range(d.nclasses):
        idx_cl = np.flatnonzero(d.labels[cl,:])
        ninstances = len(idx_cl)

        groups_to_go = groups_per_class
        while groups_to_go > 0:
            ngroups = min(int(ninstances / group_size), groups_to_go)

            idx = idx_cl[np.random.permutation(ninstances)[:ngroups*group_size]].reshape(group_size,-1)

            data = d.data[:,:,idx]
            if mean:
                data = np.mean(d.data[:,:,idx], axis=2)

            if mean:
                feat_dim_lab = d.feat_dim_lab
            else:
                feat_dim_lab = d.feat_dim_lab + ['trials'] if d.feat_dim_lab else None

            if mean:
                feat_lab = d.feat_lab
            else:
                feat_lab = d.feat_lab + [range(group_size)] if d.feat_lab else None

            labels = d.labels[:,idx[0,:]]
            ids = d.ids[:,idx[0,:]]

            d_grouped = DataSet(
                data=data, labels=labels, ids=ids, feat_dim_lab=feat_dim_lab,
                feat_lab=feat_lab, default = d
            )

            d_trials.append(d_grouped)
            idxs.append(idx)

            groups_to_go -= ngroups

    if len(d_trials) == 0:
        return ([], [])
    else:
        return (concatenate(d_trials, ignore_index=True), np.hstack(idxs))

def reject_trials(d, cutoff=100, time_range=None):
    '''
    Reject trials by thresholding the absolute amplitude. 

    Parameters
    ----------

    d : :class:`DataSet`
        The dataset to filter.
    cutoff : float (default=100)
        Any trials with a feature larger than this value are rejected.
        Alternatively, a list containing cutoff values for each channel can be
        specified. 
    time_range : tuple (default=all samples)
        Range (begin, end) for which to apply the thresholding. Values are
        given as sample indices.

    Returns
    -------
    d : :class:`DataSet`
        Filtered dataset.
    reject : :class:`numpy.Array`
        Boolean mask used to reject indices.
    '''
    if time_range == None:
        time_range = (0, d.data.shape[1])

    if hasattr(cutoff, '__iter__'):
        nchannels = d.data.shape[0]
        assert len(cutoff) == nchannels

        reject = np.any(
                [np.any(np.abs(d.data[i,time_range[0]:time_range[1],...]) > cutoff[i], axis=0)
                for i in range(nchannels)],
            axis=0)
    else:
        reject = np.any(np.any(np.abs(d.data[:,time_range[0]:time_range[1],...]) > cutoff, axis=0), axis=0)

    reject = np.logical_not(reject)

    return (d[reject], reject)

def slice(d, markers_to_class, offsets):
    '''
    Slice function, used to extract fixed-length segments (called trials) of
    EEG from a recording. Opposite of :func:`psychic.concatenate_trials`.
    Segments are sliced based on the onset of some event code.

    Given for example an EEG recording which contains two marker codes:

    1. Left finger tapping
    2. Right finger tapping

    Trials can be extracted in the following manner:

    >>> import psychic
    >>> d = psychic.load_bdf(psychic.find_data_path('priming-short.bdf'))
    >>> mdict = {1:'related', 2:'unrelated'}
    >>> sample_rate = psychic.get_samplerate(d)
    >>> begin = int(-0.2 * sample_rate)
    >>> end = int(1.0 * sample_rate)
    >>> trials = psychic.slice(d, mdict, (begin, end))
    >>> print trials
    DataSet with 208 instances, 12280 features [40x307], 2 classes: [104, 104], extras: []
     
    Parameters
    ----------
    markers_to_class : dict
        A dictionary containing as keys the event codes to use as onset of the
        trial and as values a class label for the resulting trials. For example
        ``{1:'left finger tapping', 2:'right finger tapping'}``
    offsets : tuple
        Indicates the time (start, end), relative to the onset of marker, to
        extract as trial. Values are given in samples.

    Returns
    -------
    d : :class:`DataSet`
        The extracted segments:

        - ``d.data``: [channels x samples x trials]
        - ``d.labels``: [classes x trials]
        - ``d.ids``: Timestamps indicating the marker onsets
        - ``d.cl_lab``: The class labels as specified in the
          ``markers_to_class`` dictionary
        - ``d.feat_lab``: Feature labels for the axes [channels (strings),
          time in seconds (floats)]
    '''
    assert len(d.feat_shape) == 1
    assert d.labels.shape[0] == 1 and d.labels.dtype == np.int 
    start_off, end_off = offsets
    data, labels, ids = [], [], []
    
    cl_lab = sorted(set(markers_to_class.values()))
    events, events_i, events_d = markers.markers_to_events(d.labels.flat)
    for (mark, cl) in markers_to_class.items():
        cl_i = cl_lab.index(cl)
        for i in events_i[events==mark]: # fails if there is *ONE* event
            (start, end) = i + start_off, i + end_off
            if start < 0 or end > d.ninstances:
                logging.getLogger('psychic.utils.slice').warning(
                    'Cannot extract slice [%d, %d] for class %s'
                    % (start, end, cl))
                continue
            dslice = d[start:end]
            data.append(dslice.data)
            labels.append(cl_i)
            ids.append(d.ids[:,i])
    
    event_time = np.arange(start_off, end_off) / float(utils.get_samplerate(d))
    feat_lab = [d.feat_lab[0], event_time.tolist()]

    if len(data) == 0:
        data = np.zeros(d.feat_shape + (len(event_time),0))
        labels = np.zeros((len(cl_lab),0))
        ids = np.zeros((1,0))
    else:
        data = np.concatenate([x[...,np.newaxis] for x in data], axis=2)
        labels = helpers.to_one_of_n(labels, class_rows=range(len(cl_lab)))
        ids = np.atleast_2d(np.vstack(ids).T)

    feat_dim_lab = ['channels', 'time']

    d = DataSet(
        data=data,
        labels=labels,
        ids=ids,
        cl_lab=cl_lab, 
        feat_lab=feat_lab, 
        feat_dim_lab=feat_dim_lab,
        default=d
    )
    return d.sorted()

def concatenate_trials(d):
    '''
    Concatenate trials into a single stream of EEG. Opposite of
    :func:`psychic.slice`.

    Parameters
    ----------
    d : :class:`DataSet`
        Some sliced data.

    Returns
    -------
    d : :class:`DataSet`
        A version of the data where the slices are concatenated:

        - ``d.data``: [channels x samples]
        - ``d.labels``: A reconstructed marker stream. All zeros except for the
          onsets of the trials, where it contains the marker code indicating
          the class of the trial. This simulated the data as it would be a
          continuous recording.
        - ``d.ids``: Timestamps for each sample
        - ``d.feat_lab``: This is set to ``d.feat_lab[0]``, which usually
          contains the channel names.
    '''
    assert d.data.ndim == 3, 'Expecting sliced data'

    nchannels = d.data.shape[0]
    trial_length = d.data.shape[1]
    ninstances = trial_length * d.ninstances
    
    data = np.transpose(d.data, [0,2,1]).reshape((nchannels, -1))
    labels = np.zeros((1, ninstances))
    labels[:, np.arange(0, ninstances, trial_length)] = [np.flatnonzero(d.labels[:,i])[0] + 1 for i in range(d.ninstances)]
    ids = np.atleast_2d( np.arange(ninstances) * np.median(np.diff([float(x) for x in d.feat_lab[1]])) )
    feat_lab = [d.feat_lab[0]]

    return DataSet(data=data, labels=labels, ids=ids, feat_lab=feat_lab)

def trial_specgram(d, samplerate=None, NFFT=256):
    '''
    Calculate a spectrogram for each channel and each trial. 

    Parameters
    ----------

    d : :class:`DataSet`
        The trials.
    samplerate : float
        The sample rate of the data. When omitted,
        :func:`psychic.get_samplerate` is used to estimate the sample rate.
    NFFT : int
        Number of FFT points to use to calculate the spectrograms.

    Returns
    -------
    d : :class:`DataSet`
        The spectrograms:

        - ``d.data``: [channels x freqs x samples x trials]
        - ``d.feat_lab``: The feature labels for the axes [channels
          (strings), frequencies (floats), time in seconds (floats)]

    '''
    assert d.data.ndim == 3

    if samplerate == None:
        assert d.feat_lab != None, 'Must either supply samplerate or feat_lab to deduce it.'
        samplerate = np.round(1./np.median(np.diff([float(x) for x in d.feat_lab[1]])))

    all_TFs = []
    for trial in range(d.ninstances):
        channel_TFs = []
        for channel in range(d.data.shape[0]):
            TF, freqs, times = specgram(d.data[channel,:,trial], NFFT, samplerate, noverlap=NFFT/2)
            channel_TFs.append(TF.T[np.newaxis,:,:])

        all_TFs.append(np.concatenate(channel_TFs, axis=0)[..., np.newaxis])
    all_TFs = np.concatenate(all_TFs, axis=3)

    nchannels, nfreqs, nsamples, ninstances = all_TFs.shape
    feat_lab = [d.feat_lab[0], times.tolist(), freqs.tolist()]
    feat_dim_lab=['channels', 'time', 'frequencies']

    return DataSet(
        data=all_TFs,
        feat_lab=feat_lab,
        feat_dim_lab=feat_dim_lab,
        default=d)

def align(trials, window, offsets):
    sample_rate = utils.get_samplerate(trials)
    w = np.arange(int(window[0]*sample_rate), int(sample_rate))
    feat_lab = [ trials.feat_lab[0],
                    (w / float(sample_rate)).tolist()]
    feat_shape = (trials.feat_shape[0], len(w))

    data = np.zeros(feat_shape + (trials.ninstances,))
    for i,t in enumerate(offsets):
        t -= window[0]
        data[:,:,i] = trials.data[:,int(t*sample_rate)+w,i]

    trials_aligned = DataSet(data=data, feat_lab=feat_lab, default=trials)

    return trials_aligned

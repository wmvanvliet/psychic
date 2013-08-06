import golem
import numpy as np
import scipy
import logging
import psychic
from matplotlib.mlab import specgram

def baseline(data, baseline_period=None):
    '''
    Remove the mean calculated over a certain period from the data.
    '''
    if baseline_period:
        assert len(baseline_period) == 2, 'Specify a begin and end point for the baseline period (in samples)'
    else:
        baseline_period = (0, data.ninstances)

    assert data.ndX.ndim <= 3

    if data.ndX.ndim == 2:
        num_samples = data.ninstances
        X = data.X - np.tile( np.mean(data.X[:,baseline_period[0]:baseline_period[1]], axis=1).T, (num_samples, 1) ).T

    else:
        num_samples = data.ndX.shape[1]
        ndX = np.zeros(data.ndX.shape, dtype=data.X.dtype)

        for i in range(data.ninstances):
            ndX[:,:,i] = data.ndX[:,:,i] - np.tile( np.mean(data.ndX[:,baseline_period[0]:baseline_period[1],i], axis=1).T, (num_samples, 1) ).T
        X = ndX.reshape(data.X.shape)

    return golem.DataSet(X=X, default=data)

def erp(data, n=0, classes=None, enforce_equal_n=True):
    '''
    For each class, calculate the Event Related Potential by averaging the 
    corresponding trials.
    '''
    assert data.ndX.ndim > 2

    if classes == None or len(classes) == 0:
        # Take all classes with >0 instances
        classes = [cl for cl in range(data.nclasses) if data.ninstances_per_class[cl] > 0]
    assert len(classes) > 0, 'No valid classes specified and no classes found with >0 instances'

    num_trials = np.min( np.array(data.ninstances_per_class)[classes] )
    assert num_trials > 0, 'For one or more classes there are no instances!'

    # Calculate ERP
    erp = np.zeros(data.ndX.shape[:-1] + (len(classes),))
    for i,cl in enumerate(classes):
        trials = data.get_class(cl).ndX

        if enforce_equal_n:
            # Enforce an equal number of trials for all classes. Picking them at random.
            # Otherwise the ERPs will be skewed, simply because a different number of trials are averaged.
            idx = range(trials.shape[-1])[:num_trials]
            np.random.shuffle(idx)
            erp[...,i] = np.mean(trials[...,idx], axis=trials.ndim-1)
        else:
            erp[...,i] = np.mean(trials, axis=trials.ndim-1)

    X = erp.reshape(-1, len(classes))
    Y = golem.helpers.to_one_of_n(classes).astype(np.bool)
    I = np.atleast_2d(classes)
    feat_shape = data.ndX.shape[:-1]
    cl_lab = [lab for i,lab in enumerate(data.cl_lab) if i in classes]

    return golem.DataSet(X=X, Y=Y, I=I, feat_shape=feat_shape, cl_lab=cl_lab, default=data)

def ttest(data, classes=None, shuffle=True):
    '''
    Calculate ttests between classes for each channel and each sample.
    '''
    assert data.nd_xs.ndim == 3
    assert data.nclasses >= 2, ('Data must contain at least two classes ',
                                'otherwise there is nothing to compare.')

    if classes == None:
        classes = [0,1]

    num_trials = np.min( np.array(data.ninstances_per_class)[classes] )

    c1 = data.ndX[..., data.Y[classes[0],:].astype(np.bool)]
    if shuffle:
        np.random.shuffle(c1)
    c1 = c1[..., :num_trials]

    c2 = data.ndX[..., data.Y[classes[1],:].astype(np.bool)]
    if shuffle:
        np.random.shuffle(c2)
    c2 = c2[..., :num_trials]

    return scipy.stats.ttest_ind(c1, c2, axis=2)

def random_groups(d, size):
    """ For each class, form groups of random trials of the given size """

    d_trials = None
    idxs = []

    for cl in range(d.nclasses):
        idx_cl = np.flatnonzero(d.Y[cl,:])
        ninstances = len(idx_cl)
        ngroups = int(ninstances / size)

        idx = idx_cl[np.random.permutation(ninstances)[:ngroups*size]].reshape(size,-1)
        d_grouped = golem.DataSet(
            X = d.ndX[:,:,idx].reshape(-1, ngroups),
            Y = d.Y[:,idx[0,:]],
            I = d.I[:,idx[0,:]],
            feat_shape = d.feat_shape + (size,),
            feat_dim_lab = d.feat_dim_lab + ['trials'] if d.feat_dim_lab else None,
            feat_nd_lab = d.feat_nd_lab + [range(size)] if d.feat_nd_lab else None,
            default = d
        )

        if d_trials == None:
            d_trials = d_grouped
        else:
            d_trials += d_grouped

        idxs.append(idx)

    return (d_trials, np.hstack(idxs))

def reject_trials(d, cutoff=100, range=None):
    '''
    Reject trials by thresholding the maximum amplitude. 
    TODO: Make this function automatically detect outliers

    required parameters:
    d: The dataset to filter

    optional parameters:
    cutoff: Any trials with a feature larger than this value are rejected
            [100]
    range:  (begin,end) Range along the 2nd dimension (samples) for which 
            to apply the thresholding [all samples].
    '''

    if range == None:
        range = (0, d.ndX.shape[1])

    reject = np.any(
               np.any(
                 np.abs(d.ndX[:,range[0]:range[1],:]) > cutoff,
                 axis=0),
               axis=0)

    return d[np.logical_not(reject)]

def slice(d, markers_to_class, offsets):
    '''
    Slice function, used to extract fixed-length segments of EEG from a recording.
    Returns [channel x frames x segment]
    '''
    assert len(d.feat_shape) == 1
    assert d.nclasses == 1
    start_off, end_off = offsets
    X, Y, I = [], [], []
    
    feat_shape = d.feat_shape + (end_off - start_off,)
    
    cl_lab = sorted(set(markers_to_class.values()))
    events, events_i, events_d = psychic.markers_to_events(d.Y.flat)
    for (mark, cl) in markers_to_class.items():
        cl_i = cl_lab.index(cl)
        for i in events_i[events==mark]: # fails if there is *ONE* event
            (start, end) = i + start_off, i + end_off
            if start < 0 or end > d.ninstances:
                logging.getLogger('psychic.utils.slice').warning(
                    'Cannot extract slice [%d, %d] for class %s' % (start, end, cl))
                continue
            dslice = d[start:end]
            X.append(dslice.X)
            Y.append(cl_i)
            I.append(d.I[:,i])
    
    ninstances = len(X)
    ndX = np.concatenate([x[...,np.newaxis] for x in X], axis=2)
    Y = golem.helpers.to_one_of_n(Y, class_rows=range(len(cl_lab)))
    I = np.atleast_2d(np.hstack(I))
    
    event_time = np.arange(start_off, end_off) / float(psychic.get_samplerate(d))
    feat_nd_lab = [d.feat_lab if d.feat_lab else ['f%d' % i for i in range(d.nfeatures)], event_time.tolist()]
    feat_dim_lab = ['channels', 'time']
    d = golem.DataSet(X=ndX.reshape(-1, ninstances), Y=Y, I=I, cl_lab=cl_lab, 
    feat_shape=feat_shape, feat_nd_lab=feat_nd_lab, 
    feat_dim_lab=feat_dim_lab, default=d)
    return d.sorted()

def concatenate_trials(d):
    '''
    Concatenate trials into a single stream of EEG. Opposite of slice.
    Returns [channel x frames]
    '''
    assert d.ndX.ndim == 3, 'Expecting sliced data'

    nchannels = d.ndX.shape[0]
    trial_length = d.ndX.shape[1]
    ninstances = trial_length * d.ninstances
    
    X = np.transpose(d.ndX, [0,2,1]).reshape((nchannels, -1))
    Y = np.zeros((1, ninstances))
    Y[:, np.arange(0, ninstances, trial_length)] = [np.flatnonzero(d.Y[:,i])[0] + 1 for i in range(d.ninstances)]
    I = np.atleast_2d( np.arange(ninstances) * np.median(np.diff([float(x) for x in d.feat_nd_lab[1]])) )
    feat_lab = d.feat_nd_lab[0]

    return golem.DataSet(X=X, Y=Y, I=I, feat_lab=feat_lab)

def trial_specgram(d, samplerate=None, NFFT=256):
    assert d.ndX.ndim == 3

    if samplerate == None:
        assert d.feat_nd_lab != None, 'Must either supply samplerate or feat_nd_lab to deduce it.'
        samplerate = np.round(1./np.median(np.diff([float(x) for x in d.feat_nd_lab[1]])))
        print 'samplerate: %.02f' % samplerate

    all_TFs = []
    for trial in range(d.ninstances):
        channel_TFs = []
        for channel in range(d.ndX.shape[0]):
            TF, freqs, times = specgram(d.ndX[channel,:,trial], NFFT, samplerate, noverlap=NFFT/2)
            channel_TFs.append(TF[np.newaxis,:,:])

        all_TFs.append(np.concatenate(channel_TFs, axis=0)[..., np.newaxis])
    all_TFs = np.concatenate(all_TFs, axis=3)

    nchannels, nfreqs, nsamples, ninstances = all_TFs.shape
    feat_shape = (nchannels, nfreqs, nsamples)
    feat_nd_lab = [d.feat_nd_lab[0],
                   ['%f' % f for f in freqs],
                   ['%f' % t for t in times],
                  ]
    feat_dim_lab=['channels', 'frequencies', 'time']

    return golem.DataSet(
        X = all_TFs.reshape(-1, ninstances),
        feat_shape=feat_shape,
        feat_nd_lab=feat_nd_lab,
        feat_dim_lab=feat_dim_lab,
        default=d)

    




=======
﻿import golem
import numpy as np
import scipy
import logging
import psychic
from matplotlib.mlab import specgram

def baseline(data, baseline_period=None):
    '''
    Remove the mean calculated over a certain period from the data.
    '''
    if baseline_period:
        assert len(baseline_period) == 2, 'Specify a begin and end point for the baseline period (in samples)'
    else:
        baseline_period = (0, data.ninstances)

    assert data.ndX.ndim <= 3

    if data.ndX.ndim == 2:
        num_samples = data.ninstances
        X = data.X - np.tile( np.mean(data.X[:,baseline_period[0]:baseline_period[1]], axis=1).T, (num_samples, 1) ).T

    else:
        num_samples = data.ndX.shape[1]
        ndX = np.zeros(data.ndX.shape, dtype=data.X.dtype)

        for i in range(data.ninstances):
            ndX[:,:,i] = data.ndX[:,:,i] - np.tile( np.mean(data.ndX[:,baseline_period[0]:baseline_period[1],i], axis=1).T, (num_samples, 1) ).T
        X = ndX.reshape(data.X.shape)

    return golem.DataSet(X=X, default=data)

def erp(data, n=0, classes=None, enforce_equal_n=True):
    '''
    For each class, calculate the Event Related Potential by averaging the 
    corresponding trials.
    '''
    assert data.ndX.ndim > 2

    if classes == None or len(classes) == 0:
        # Take all classes with >0 instances
        classes = [cl for cl in range(data.nclasses) if data.ninstances_per_class[cl] > 0]
    assert len(classes) > 0, 'No valid classes specified and no classes found with >0 instances'

    num_trials = np.min( np.array(data.ninstances_per_class)[classes] )
    assert num_trials > 0, 'For one or more classes there are no instances!'

    # Calculate ERP
    erp = np.zeros(data.ndX.shape[:-1] + (len(classes),))
    for i,cl in enumerate(classes):
        trials = data.get_class(cl).ndX

        if enforce_equal_n:
            # Enforce an equal number of trials for all classes. Picking them at random.
            # Otherwise the ERPs will be skewed, simply because a different number of trials are averaged.
            idx = range(trials.shape[-1])[:num_trials]
            np.random.shuffle(idx)
            erp[...,i] = np.mean(trials[...,idx], axis=trials.ndim-1)
        else:
            erp[...,i] = np.mean(trials, axis=trials.ndim-1)

    X = erp.reshape(-1, len(classes))
    Y = golem.helpers.to_one_of_n(classes).astype(np.bool)
    I = np.atleast_2d(classes)
    feat_shape = data.ndX.shape[:-1]
    cl_lab = [lab for i,lab in enumerate(data.cl_lab) if i in classes]

    return golem.DataSet(X=X, Y=Y, I=I, feat_shape=feat_shape, cl_lab=cl_lab, default=data)

def ttest(data, classes=None, shuffle=True):
    '''
    Calculate ttests between classes for each channel and each sample.
    '''
    assert data.nd_xs.ndim == 3
    assert data.nclasses >= 2, ('Data must contain at least two classes ',
                                'otherwise there is nothing to compare.')

    if classes == None:
        classes = [0,1]

    num_trials = np.min( np.array(data.ninstances_per_class)[classes] )

    c1 = data.ndX[..., data.Y[classes[0],:].astype(np.bool)]
    if shuffle:
        np.random.shuffle(c1)
    c1 = c1[..., :num_trials]

    c2 = data.ndX[..., data.Y[classes[1],:].astype(np.bool)]
    if shuffle:
        np.random.shuffle(c2)
    c2 = c2[..., :num_trials]

    return scipy.stats.ttest_ind(c1, c2, axis=2)

def random_groups(d, size):
    """ For each class, form groups of random trials of the given size """

    d_trials = None
    idxs = []

    for cl in range(d.nclasses):
        d_cl = d.get_class(cl)
        ngroups = int(d_cl.ninstances / size)

        idx = np.random.permutation(d_cl.ninstances)[:ngroups*size].reshape(size,-1)
        d_grouped = golem.DataSet(
            X = d_cl.ndX[:,:,idx].reshape(-1, ngroups),
            Y = d_cl.Y[:,idx[0,:]],
            I = d_cl.I[:,idx[0,:]],
            feat_shape = d_cl.feat_shape + (size,),
            feat_dim_lab = d_cl.feat_dim_lab + ['trials'] if d_cl.feat_dim_lab else None,
            feat_nd_lab = d_cl.feat_nd_lab + [range(size)] if d_cl.feat_nd_lab else None,
            default = d_cl
        )

        if d_trials == None:
            d_trials = d_grouped
        else:
            d_trials += d_grouped

        idxs.append(idx)

    return (d_trials, np.hstack(idxs))

def reject_trials(d, cutoff=100, range=None):
    '''
    Reject trials by thresholding the maximum amplitude. 
    TODO: Make this function automatically detect outliers

    required parameters:
    d: The dataset to filter

    optional parameters:
    cutoff: Any trials with a feature larger than this value are rejected
            [100]
    range:  (begin,end) Range along the 2nd dimension (samples) for which 
            to apply the thresholding [all samples].
    '''

    if range == None:
        range = (0, d.ndX.shape[1])

    reject = np.any(
               np.any(
                 np.abs(d.ndX[:,range[0]:range[1],:]) > cutoff,
                 axis=0),
               axis=0)

    return d[np.logical_not(reject)]

def slice(d, markers_to_class, offsets):
    '''
    Slice function, used to extract fixed-length segments of EEG from a recording.
    Returns [channel x frames x segment]
    '''
    assert len(d.feat_shape) == 1
    assert d.nclasses == 1
    start_off, end_off = offsets
    X, Y, I = [], [], []
    
    feat_shape = d.feat_shape + (end_off - start_off,)
    
    cl_lab = sorted(set(markers_to_class.values()))
    events, events_i, events_d = psychic.markers_to_events(d.Y.flat)
    for (mark, cl) in markers_to_class.items():
        cl_i = cl_lab.index(cl)
        for i in events_i[events==mark]: # fails if there is *ONE* event
            (start, end) = i + start_off, i + end_off
            if start < 0 or end > d.ninstances:
                logging.getLogger('psychic.utils.slice').warning(
                    'Cannot extract slice [%d, %d] for class %s' % (start, end, cl))
                continue
            dslice = d[start:end]
            X.append(dslice.X)
            Y.append(cl_i)
            I.append(d.I[:,i])
    
    ninstances = len(X)
    ndX = np.concatenate([x[...,np.newaxis] for x in X], axis=2)
    Y = golem.helpers.to_one_of_n(Y, class_rows=range(len(cl_lab)))
    I = np.atleast_2d(np.hstack(I))
    
    event_time = np.arange(start_off, end_off) / float(psychic.get_samplerate(d))
    feat_nd_lab = [d.feat_lab if d.feat_lab else ['f%d' % i for i in
        range(d.nfeatures)], event_time.tolist()]
    feat_dim_lab = ['channels', 'time']
    d = golem.DataSet(X=ndX.reshape(-1, ninstances), Y=Y, I=I, cl_lab=cl_lab, 
    feat_shape=feat_shape, feat_nd_lab=feat_nd_lab, 
    feat_dim_lab=feat_dim_lab, default=d)
    return d.sorted()

def concatenate_trials(d):
    '''
    Concatenate trials into a single stream of EEG. Opposite of slice.
    Returns [channel x frames]
    '''
    assert d.ndX.ndim == 3, 'Expecting sliced data'

    nchannels = d.ndX.shape[0]
    trial_length = d.ndX.shape[1]
    ninstances = trial_length * d.ninstances
    
    X = np.transpose(d.ndX, [0,2,1]).reshape((nchannels, -1))
    Y = np.zeros((1, ninstances))
    Y[:, np.arange(0, ninstances, trial_length)] = [np.flatnonzero(d.Y[:,i])[0] + 1 for i in range(d.ninstances)]
    I = np.atleast_2d( np.arange(ninstances) * np.median(np.diff([float(x) for x in d.feat_nd_lab[1]])) )
    feat_lab = d.feat_nd_lab[0]

    return golem.DataSet(X=X, Y=Y, I=I, feat_lab=feat_lab)

def trial_specgram(d, samplerate=None, NFFT=256):
    assert d.ndX.ndim == 3

    if samplerate == None:
        assert d.feat_nd_lab != None, 'Must either supply samplerate or feat_nd_lab to deduce it.'
        samplerate = np.round(1./np.median(np.diff([float(x) for x in d.feat_nd_lab[1]])))
        print 'samplerate: %.02f' % samplerate

    all_TFs = []
    for trial in range(d.ninstances):
        channel_TFs = []
        for channel in range(d.ndX.shape[0]):
            TF, freqs, times = specgram(d.ndX[channel,:,trial], NFFT, samplerate, noverlap=NFFT/2)
            channel_TFs.append(TF[np.newaxis,:,:])

        all_TFs.append(np.concatenate(channel_TFs, axis=0)[..., np.newaxis])
    all_TFs = np.concatenate(all_TFs, axis=3)

    nchannels, nfreqs, nsamples, ninstances = all_TFs.shape
    feat_shape = (nchannels, nfreqs, nsamples)
    feat_nd_lab = [d.feat_nd_lab[0],
                   ['%f' % f for f in freqs],
                   ['%f' % t for t in times],
                  ]
    feat_dim_lab=['channels', 'frequencies', 'time']

    return golem.DataSet(
        X = all_TFs.reshape(-1, ninstances),
        feat_shape=feat_shape,
        feat_nd_lab=feat_nd_lab,
        feat_dim_lab=feat_dim_lab,
        default=d)

    





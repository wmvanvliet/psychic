import golem
import psychic
import numpy as np
import scipy
import logging
import inspect

def baseline(data, baseline_period=None):
    if baseline_period:
        assert len(baseline_period) == 2, 'Specify a begin and end point for the baseline period (in samples)'
    else:
        baseline_period = (0, data.ninstances)

    assert data.nd_xs.ndim <= 3

    if data.nd_xs.ndim == 2:
        num_samples = data.ninstances
        xs = data.xs - np.tile( np.mean(xs[baseline_period[0]:baseline_period[1],:], axis=0), (num_samples, 1) )

    else:
        num_samples = data.nd_xs.shape[1]
        xs = np.zeros(data.nd_xs.shape, dtype=data.xs.dtype)

        for i in range(data.nd_xs.shape[0]):
            xs[i,:,:] = data.nd_xs[i,:,:] - np.tile( np.mean(data.nd_xs[i,baseline_period[0]:baseline_period[1],:], axis=0), (num_samples, 1) )

    return golem.DataSet(xs=xs.reshape(data.xs.shape), default=data)

def erp(data, n=0, classes=None, enforce_equal_n=True, n_offset=0):
    assert data.nd_xs.ndim == 3

    if classes == None or len(classes) == 0:
        # Take all classes with >0 instances
        classes = [cl for cl in range(data.nclasses) if data.ninstances_per_class[cl] > 0]
    assert len(classes) > 0, 'No valid classes specified and no classes found with >0 instances'

    num_trials = np.min( np.array(data.ninstances_per_class)[classes] )
    assert num_trials > 0, 'For one or more classes there are no instances!'

    num_samples = data.nd_xs.shape[1]
    num_channels = data.nd_xs.shape[2]

    # Calculate ERP
    erp = np.zeros( (len(classes), num_samples, num_channels) )
    for i,cl in enumerate(classes):
        trials = data.get_class(cl).nd_xs

        if n > 0:
            trials = trials[n_offset:n_offset+n,:,:]

        if enforce_equal_n:
            # Enforce an equal number of trials for all classes. Picking them at random.
            # Otherwise the ERPs will be skewed, simply because a different number of trials are averaged.
            idx = range(trials.shape[0])[:num_trials]
            np.random.shuffle(idx)
            erp[i,:,:] = np.mean(trials[idx,:,:], axis=0)
        else:
            erp[i,:,:] = np.mean(trials, axis=0)

    xs = erp.reshape(len(classes), -1)
    ys = np.atleast_2d(classes).T
    ids = np.atleast_2d(classes).T
    feat_shape = (num_samples, num_channels)
    cl_lab = []

    return golem.DataSet(xs=xs, ys=ys, ids=ids, feat_shape=feat_shape, cl_lab=cl_lab, default=data)

def ttest(data, classes=None, shuffle=True):
    assert data.nd_xs.ndim == 3
    assert data.nclasses >= 2, 'Data must contain two classes, otherwise there is nothing to compare.'

    if classes == None:
        classes = [0,1]

    num_trials = np.min( np.array(data.ninstances_per_class)[classes] )

    c1 = data.nd_xs[data.ys[:, classes[0]].astype(np.bool),:,:]
    if shuffle:
        np.random.shuffle(c1)
    c1 = c1[:num_trials,:,:]

    c2 = data.nd_xs[data.ys[:, classes[1]].astype(np.bool),:,:]
    if shuffle:
        np.random.shuffle(c2)
    c2 = c2[:num_trials,:,:]

    return scipy.stats.ttest_ind(c1, c2, axis=0)

def random_groups(d, size):
    """ For each class, form groups of random trials of the given size """

    d_trials = None
    idxs = []

    for cl in range(d.nclasses):
        d_cl = d.get_class(cl)
        ngroups = int(d_cl.ninstances / size)

        idx = np.random.permutation(d_cl.ninstances)[:ngroups*size].reshape(size,-1)
        print idx
        d_grouped = golem.DataSet(
            X = d_cl.ndX[:,:,idx].reshape(-1, ngroups),
            Y = d_cl.Y[:,idx[0,:]],
            I = d_cl.I[:,idx[0,:]],
            feat_shape = d_cl.feat_shape + (size,),
            feat_dim_lab = d_cl.feat_dim_lab + ['trials'],
            feat_nd_lab = d_cl.feat_nd_lab + [range(size)],
            default = d_cl
        )

        if d_trials == None:
            d_trials = d_grouped
        else:
            d_trials += d_grouped

        idxs.append(idx)

    return (d_trials, np.hstack(idxs))

def reject_trials(d, cutoff=100, range=None):
    """
    Reject trials by thresholding the maximum amplitude. 
    TODO: Make this function automatically detect outliers

    required parameters:
    d: The dataset to filter

    optional parameters:
    cutoff: Any trials with a feature larger than this value are rejected
            [100]
    range:  [begin:end] Range along the 2nd dimension (samples) for which 
            to apply the thresholding
    """

    if range == None:
        range = (0, d.ndX.shape[1])

    reject = np.any(
               np.any(
                 np.abs(d.ndX[:,range[0]:range[1],:]) > cutoff,
                 axis=0),
               axis=0)

    return d[np.logical_not(reject)]

from ..dataset import DataSet
from . import BaseNode
import numpy as np
from ..trials import reject_trials, baseline, erp, random_groups, ungroup
import inspect

class Mean(BaseNode):
    """ Take the mean along a given axis """
    def __init__(self, axis=2, n=None):
        BaseNode.__init__(self)
        self.axis=axis
        self.n = n

    def apply_(self, d):
        if self.n is None:
            self.n = d.data.shape[2]
        data = np.mean(d.data[:,:,:self.n], axis=self.axis)
        feat_dim_lab = [lab for i,lab in enumerate(d.feat_dim_lab) if i != self.axis]
        return DataSet(data=data, feat_dim_lab=feat_dim_lab, default=d)

class ERP(BaseNode):
    '''
    For each class, calculate the Event Related Potential by averaging the 
    corresponding trials. Note: no baselining is performed, see
    :class:`psychic.nodes.Baseline`.

    Parameters
    ----------
    classes: list (optional)
        When specified, the ERP is only calculated for the classes with the
        given indices.
    enforce_equal_n : bool (default=True)
        When set, each ERP is calculated by averaging the same number of
        trials. For example, if class1 has m and class2 has n trials and m > n.
        The ERP for class1 will be calculated by taking n random trials from
        class1.
    axis : int (default=2)
        Axis along which to take the mean. Defaults to 2, which normally
        correspond to averaging across trials.
    '''

    def __init__(self, classes=None, enforce_equal_n=True, axis=2):
        BaseNode.__init__(self)
        self.classes = classes
        self.enforce_equal_n = enforce_equal_n
        self.axis = axis

    def train_(self, d):
        if self.classes is None:
            self.classes = list(range(d.nclasses))

    def apply_(self, d):
        return erp(d, classes=self.classes, enforce_equal_n=self.enforce_equal_n)

class Blowup(BaseNode):
    """ Blow up dataset """

    def __init__(self, num_combinations=1000, enable_during_application=False,
            mean=False):
        BaseNode.__init__(self)
        self.num_combinations = num_combinations
        self.enable_during_application = enable_during_application
        self.mean = mean

    def apply_(self, d):
        if d.ninstances == 0:
            return d

        if (not self.enable_during_application) and inspect.stack()[3][3] == 'apply_':
            self.log.debug('Not blowing up data in application mode.')
            return d

        assert d.data.ndim == 4

        classes = list(range(d.nclasses))
        cl_lab = d.cl_lab
        if np.min( np.array(d.ninstances_per_class) ) == 0:
            self.log.warning("Some classes have 0 instances and will be skipped")
            classes = [cl for cl in classes if d.ninstances_per_class[cl] > 0]
            cl_lab = [lab for cl,lab in enumerate(cl_lab) if d.ninstances_per_class[cl] > 0]

        num_combinations = self.num_combinations
        num_repetitions = d.data.shape[2]

        d2 = ungroup(d, axis=2)
        d2, self.reverse_idx = random_groups(d2, num_repetitions,
                num_combinations, self.mean)

        return d2

class RejectTrials(BaseNode):
    """
    Node that rejects trials which features that exceed a certain threshold.
    Wrapper around psychic.reject_trials()
    """
    def __init__(self, cutoff=100, std_cutoff=False, range=None):
        BaseNode.__init__(self)
        self.cutoff = cutoff
        self.std_cutoff = std_cutoff
        self.range = range
        self.trained = False

    def train_(self, d):
        nchannels = d.data.shape[0]
        if self.std_cutoff:
            self.channel_stds = np.array([np.std(d.data[i,...]) for i in range(nchannels)])
            self.cutoff = self.cutoff * self.channel_stds
        self.trained = True

    def apply_(self, d):
        assert (not self.std_cutoff or self.trained), \
            'When specifying the cutoff in terms of standard deviations, training of the node is mandatory.'

        d, self.reject_mask = reject_trials(d, self.cutoff, self.range)
        self.log.info('Rejected %d trials' %
                len(np.flatnonzero(np.logical_not(self.reject_mask))))
        return d

class Baseline(BaseNode):
    '''
    Node that performs baselining on trials. For each channel, the baseline
    voltage is calculate by averaging over the baseline period. The baseline
    voltage is then substracted from the signal.

    The baseline period is specified as a tuple (begin, end) of two timestamps.
    These timestamps are compared with ``d.feat_lab[1]`` to determine the
    corresponding range in terms of samples. For this to work,
    ``d.feat_lab[1]`` must contain a meaningfull list of timestamps: one for
    each EEG sample. Trials extracted through :class:`psychic.nodes.Slice`` and
    :class:`psychic.nodes.SlidingWindow` contain these timestamps.

    See section :ref:`baseline` in the documentation for an example usage.

    Parameters
    ----------
    baseline_period : tuple (float, float) (default=entire trial)
        The start and end (inclusive) of the period to calculate the baseline
        over. Values are given in seconds relative to the timestamps contained
        in ``d.feat_lab[1]``.
    '''

    def __init__(self, baseline_period=None):
        BaseNode.__init__(self)
        self.baseline_period = baseline_period

    def train_(self, d):
        if self.baseline_period is None:
            self.begin_idx = 0
            self.end_idx = -1
        else:
            self.begin_idx = np.searchsorted(d.feat_lab[1], self.baseline_period[0])
            self.end_idx = np.searchsorted(d.feat_lab[1], self.baseline_period[1]) + 1

    def apply_(self, d):
        return baseline(d, [self.begin_idx, self.end_idx])

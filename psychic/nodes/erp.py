import golem
from golem.nodes import BaseNode
import numpy as np
from ..trials import reject_trials
from ..trials import baseline
from ..trials import erp
import inspect

class Mean(BaseNode):
    """ Take the mean along a given axis """
    def __init__(self, axis=2, n=None):
        BaseNode.__init__(self)
        self.axis=axis
        self.n = n

    def apply_(self, d):
        if self.n == None:
            self.n = d.ndX.shape[2]
        X = np.mean(d.ndX[:,:,:self.n], axis=self.axis).reshape(-1, d.ninstances)
        feat_shape = tuple([dim for i,dim in enumerate(d.feat_shape) if i != self.axis])
        feat_dim_lab = [lab for i,lab in enumerate(d.feat_dim_lab) if i != self.axis]
        return golem.DataSet(X=X, feat_shape=feat_shape, feat_dim_lab=feat_dim_lab, default=d)

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
        if self.classes == None:
            self.classes = range(d.nclasses)

    def apply_(self, d):
        return erp(d, classes=self.classes, enforce_equal_n=self.enforce_equal_n)

class Blowup(BaseNode):
    """ Blow up dataset """

    def __init__(self, num_combinations=1000, enable_during_application=False):
        BaseNode.__init__(self)
        self.num_combinations = num_combinations
        self.enable_during_application = enable_during_application

    def apply_(self, d):
        if d.ninstances == 0:
            return d

        if (not self.enable_during_application) and inspect.stack()[2][3] == 'apply_':
            self.log.debug('Not blowing up data in application mode.')
            return d

        assert d.ndX.ndim == 4

        classes = range(d.nclasses)
        cl_lab = d.cl_lab
        if np.min( np.array(d.ninstances_per_class) ) == 0:
            self.log.warning("Some classes have 0 instances and will be skipped")
            classes = [cl for cl in classes if d.ninstances_per_class[cl] > 0]
            cl_lab = [lab for cl,lab in enumerate(cl_lab) if d.ninstances_per_class[cl] > 0]

        num_combinations = self.num_combinations
        num_repetitions = d.ndX.shape[2]

        # For each class, generate random combinations
        xs = []
        reverse_idxs = []
        for cl_i in classes:
            combinations = []

            X = d.get_class(cl_i).ndX

            if X.shape[3] == 0:
                # No instances of the class
                continue

            # Create reverse index for bookkeeping
            reverse_idx = np.flatnonzero(d.Y[cl_i,:] == 1)
            reverse_idx *= num_repetitions
            reverse_idx = reverse_idx.repeat(num_repetitions)
            reverse_idx += np.tile(np.arange(num_repetitions), X.shape[3])

            # Unfold groups
            X = X.reshape(X.shape[:-2] + (-1,))

            # We're going to shuffle X along the 3rd axis
            idx = range(X.shape[2])

            # Use all of the trials as much as possible
            for i in range( int(num_repetitions*num_combinations/len(idx)) ):
                np.random.shuffle(idx)
                combinations.append(X[:,:,idx])
                reverse_idxs.append(reverse_idx[idx])


            # If num_combinations is not a multitude of len(idx),
            # append some extra
            to_go = (num_repetitions*num_combinations) % len(idx)
            if to_go > 0:    
                np.random.shuffle(idx)
                combinations.append(X[:,:,idx[:to_go]])
                reverse_idxs.append(reverse_idx[idx[:to_go]])

            xs.append( np.concatenate(combinations, axis=2).reshape(d.nfeatures, -1) )

        
        X = np.hstack(xs)

        # Construct Y
        Y = np.zeros(( len(classes), len(classes)*num_combinations ))
        for cl in range(len(classes)):
            offset = cl*num_combinations
            Y[cl,offset:offset + num_combinations] = 1

        # Construct I
        I=np.arange(X.shape[1])

        self.reverse_idx = np.hstack(reverse_idxs).reshape(num_repetitions, -1)
        return golem.DataSet(X=X, Y=Y, I=I, cl_lab=cl_lab, default=d)

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
        nchannels = d.ndX.shape[0]
        if self.std_cutoff:
            self.channel_stds = np.array([np.std(d.ndX[i,...]) for i in range(nchannels)])
            self.cutoff = self.cutoff * self.channel_stds
        self.trained = True

    def apply_(self, d):
        assert (not self.std_cutoff or self.trained), \
            'When specifying the cutoff in terms of standard deviations, training of the node is mandatory.'

        d, self.reject_mask = reject_trials(d, self.cutoff, self.range)
        self.log.info('Rejected %d trials' % len(np.flatnonzero(np.logical_not(self.reject_mask))))
        return d

class Baseline(BaseNode):
    '''
    Node that performs baselining on trials. For each channel, the baseline
    voltage is calculate by averaging over the baseline period. The baseline
    voltage is then substracted from the signal.

    The baseline period is specified as a tuple (begin, end) of two timestamps.
    These timestamps are compared with ``d.feat_nd_lab[1]`` to determine the
    corresponding range in terms of samples. For this to work,
    ``d.feat_nd_lab[1]`` must contain a meaningfull list of timestamps: one for
    each EEG sample. Trials extracted through :class:`psychic.nodes.Slice`` and
    :class:`psychic.nodes.SlidingWindow` contain these timestamps.

    See section :ref:`baseline` in the documentation for an example usage.

    Parameters
    ----------
    baseline_period : tuple (float, float) (default=entire trial)
        The start and end (inclusive) of the period to calculate the baseline
        over. Values are given in seconds relative to the timestamps contained
        in ``d.feat_nd_lab[1]``.
    '''

    def __init__(self, baseline_period=None):
        BaseNode.__init__(self)
        self.baseline_period = baseline_period

    def train_(self, d):
        if self.baseline_period == None:
            self.begin_idx = 0
            self.end_idx = -1
        else:
            self.begin_idx = np.searchsorted(d.feat_nd_lab[1], self.baseline_period[0])
            self.end_idx = np.searchsorted(d.feat_nd_lab[1], self.baseline_period[1]) + 1

    def apply_(self, d):
        return baseline(d, [self.begin_idx, self.end_idx])

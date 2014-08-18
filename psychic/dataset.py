import cPickle
import operator
import numpy as np
import helpers
import copy
import operator

class DataSet(object):
    """
    A data set consists of instances with features and labels, and some
    additional descriptors.
    
    Parameters
    ----------
    data : nd-array (required)
        Instances with features. Features can be multi-dimensional. For
        example, in the case of EEG, instances can be epochs, consisting of
        time samples for each channel.

    labels : 2D-array
        The true class labels (or values) for the instances. There are three
        ways of specifying labels:

        1. As a flat list assigning an integer value to each instance.
        2. As a 2D boolean array, where each row corresponds to a class and
           each column corresponds to an instance. A value of True means the
           an instance belongs to a certain class. Instances can belong to
           multiple classes at the same time.
        3. As a 3D boolean array, where each row corresponds to a class and
           each column corresponds to an instance. Each value is a score,
           describing how much a certain instance belongs to a certain class,
           for example a probability score. In summary views of the data, each
           instance will be assigned to the class with the highest score.

        For example, if for a certain sample the labels are [False, True,
        False] this means this sample belongs to the second class. The names of
        the classes are stored in ``cl_lab``.

        When omitted, each instance will be assigned to the same class.

    ids : list
        A unique identifier per instance. If not provided, it will generate a
        unique integer id from 0 to the number of instances. In the case of
        EEG, this can contain the time stamps of the samples.

    cl_lab : list
        A list of string descriptors for each class.

    feat_lab : list of lists
        For each feature dimension, a list of string feature descriptors.
        
        For example, in the case of EEG, this could be
        ``[['C3','C4'],['0.01','0.02','0.03','0.04']]``, which denotes that
        the first dimension of the data describes two channels, and the second
        dimension 4 time samples.

    feat_dim_lab : list
        For each feature dimension, a string descriptor.  In the case of EEG,
        this could be ``['channels', 'time']``.

    extra : dict
        A dictionary that can be used to store any additional information you
        may want to include.

    default : DataSet
        A default dataset from which all the information will be obtained that
        is not defined in the initialization of a new dataset.


    Attributes
    ----------
    data
    labels
    ids
    X
    y
    feat_shape
    prior
    ninstances
    ninstances_per_class
    nfeatures
    nclasses
    cl_lab : list
        A list of string descriptors for each class.
    feat_lab : list of lists
        A list of string descriptors the features along each dimension.
        For example, in the case of EEG, this could be
        ``[['C3','C4'],['0.01','0.02','0.03','0.04']]``, which denotes that the
        first dimension of the data describes two channels, and the second
        dimension 4 time samples.
    feat_dim_lab : list
        For each feature dimension, a string descriptor.
        In the case of EEG, this could be ``['channels', 'time']``.
    extra : dict
        A dictionary that can be used to store any additional information you
        may want to include.

    Notes
    -----
    For security, it is not possible to write to an already created dataset
    (``data``, ``labels``, and ``ids`` are locked). This way, you can be
    certain that a dataset will not be modified from analysis chain to another.
    
    **Handy operators** (where ``d`` is a ``DataSet``): 
    
    ``d3 = d1 + d2``
        Concatenating datasets together.
    
    ``if d1 == d2``
        Comparing datasets.
    
    ``len(d)``
        Return the number of samples in the dataset.
    
    ``d[5]``
        Return the instance with index 5.
    
    ``str(d)``
        Return a string representation of the dataset.
    
    """
    @property
    def data(self):
        '''
        Instances with features. Features can be multi-dimensional. For
        example, in the case of EEG, instances can be epochs, consisting of
        time samples for each channel.
        '''
        return self._data

    @property
    def labels(self):
        '''
        The true class labels (or values) for the instances. There are three
        ways of specifying labels:

        1. As a flat list assigning an integer value to each instance.
        2. As a 2D boolean array, where each row corresponds to a class and
           each column corresponds to an instance. A value of True means the
           an instance belongs to a certain class. Instances can belong to
           multiple classes at the same time.
        3. As a 3D boolean array, where each row corresponds to a class and
           each column corresponds to an instance. Each value is a score,
           describing how much a certain instance belongs to a certain class,
           for example a probability score. In summary views of the data, each
           instance will be assigned to the class with the highest score.

        For example, if for a certain sample the labels are [False, True,
        False] this means this sample belongs to the second class. The names of
        the classes are stored in ``cl_lab``.

        When omitted, each instance will be assigned to the same class.
        '''
        return self._labels

    @property
    def ids(self):
        '''
        A unique identifier per instance. If not provided, it will generate a
        unique integer id from 0 to the number of samples. In the case of EEG,
        this can contain the time stamps of the samples.
        '''
        return self._ids

    @property
    def X(self):
        '''
        A flattened version of `data`. This is a 2D array (features x
        instances) suitable for scikit-learn. 
        '''
        return self._data.reshape(-1, self.ninstances).T

    @property
    def y(self):
        '''
        A flat version of `labels`. This is a list of integer class labels,
        assigning each instance to a class.  This is suitable for scikit-learn.
        '''
        if self._labels.shape[0] > 1:
            return np.argmax(self._labels, axis=0)
        else:
            return self._labels.flatten()

    def __init__(self, data=None, labels=None, ids=None, cl_lab=None,
        feat_lab=None, feat_dim_lab=None, possible_labels=None, extra=None,
        default=None):
        '''
        Create a new dataset.
        '''
        # first, take care of data, labels and ids
        if default != None:
            assert isinstance(default, DataSet), 'Default is not a DataSet'

        if data == None:
            if default != None:
                data = default.data
            else:
                raise ValueError, 'No data given'
        self._data = data = np.atleast_2d(data)
 
        if labels == None:
            # Maybe labels can be copied from default parameter
            if default != None and default.labels.shape[1] == data.shape[-1]: 
                labels = default.labels

            # Assign all instances to the same class
            else:
                labels = np.ones(data.shape[-1], dtype=np.bool)

        self._labels = labels = np.atleast_2d(labels)

        # When labels are specified as a single list of integers, keep track
        # of all possible labels.
        if labels.shape[0] == 1 and labels.dtype == np.int:
            if possible_labels == None:
                # Maybe possible labels can be copied from default parameter
                if (default != None and hasattr(default, 'possible_labels') and
                  len(np.setdiff1d(np.unique(labels), default.possible_labels)) == 0):
                    self.possible_labels = default.possible_labels
                else:
                    self.possible_labels = np.unique(labels)
            else:
                self.possible_labels = np.asarray(possible_labels)

        if ids == None:
            if default != None and default.ids.shape[1] == data.shape[-1]:
                ids = default.ids
            else:
                ids = np.arange(data.shape[-1], dtype=np.int)
        self._ids = ids = np.atleast_2d(ids)

        # test essential properties
        if self.labels.ndim != 2:
            raise ValueError('Only 2D arrays are supported for labels.')
        if self.ids.ndim != 2:
            raise ValueError('Only 2D arrays are supported for ids.')
        if not (self.data.shape[-1] == self.labels.shape[1] == self.ids.shape[1]):
            raise ValueError('Number of instances (cols) does not match')
        if np.unique(self.ids[0]).size != self.ninstances:
            raise ValueError('The ids are not unique.')

        if not np.all(np.isfinite(self.data)):
            raise ValueError('Only finite values are allowed as data')
        
        # Lock data, labels and ids
        for arr in [self.data, self.labels, self.ids]:
            arr.flags.writeable = False
            
        # Add required structural info:
        if cl_lab == None:
            if default != None and len(default.cl_lab) == self.nclasses:
                self.cl_lab = default.cl_lab
            else:
                if self.labels.dtype == np.int and self.labels.shape[0] == 1:
                    self.cl_lab = ['class %d' % i for i in self.possible_labels]
                else:
                    self.cl_lab = ['class %d' % i for i in range(self.nclasses)]
        else:
            self.cl_lab = cl_lab

        feat_shape = data.shape[:-1]

        if feat_lab == None:
            if (
                default != None and
                default.feat_lab != None and
                len(feat_shape) == len(default.feat_lab) and
                np.all([len(x) == y for x,y in zip(default.feat_lab, self.feat_shape)])
            ):
                self.feat_lab = default.feat_lab
            else:
                self.feat_lab = [range(x) for x in feat_shape]
        else:
            # Make sure feat_lab is a list of lists
            if not np.all([isinstance(x, list) for x in feat_lab]):
                feat_lab = [feat_lab]
            self.feat_lab = copy.deepcopy(feat_lab)

        # Now we are basically done, but let's add optional info
        if feat_dim_lab == None:
            if(default != None and default.feat_dim_lab != None and \
                len(default.feat_dim_lab) == len(feat_shape)):
                self.feat_dim_lab = default.feat_dim_lab
            else:
                self.feat_dim_lab = ['feat_dim%d' % i for i in range(len(feat_shape))]
        else:
            self.feat_dim_lab = feat_dim_lab

        if extra == None:
            if default != None and default.extra != None:
                self.extra = default.extra
            else:
                self.extra = {}
        else:
            self.extra = extra

        self.check_consistency()

    def check_consistency(self):
        assert isinstance(self.cl_lab, list), 'cl_lab not a list'
        assert np.all([isinstance(x, list) for x in self.feat_lab]), 'feat_lab not a list of lists'
        assert isinstance(self.feat_dim_lab, list), 'feat_dim_lab not a list'
        assert isinstance(self.extra, dict), 'extra not a dict'

        if (len(self.feat_shape) != len(self.feat_lab) or
            not np.all([len(x) == y for x,y in zip(self.feat_lab, self.feat_shape)])
           ):
            raise ValueError('Dimensions of feat_lab %s do not match the those of the data %s' % 
                (repr(tuple([len(x) for x in self.feat_lab])), repr(self.feat_shape)))
        if len(self.cl_lab) != self.nclasses:
            raise ValueError('The number of class labels does not match #classes')
        if len(self.feat_shape) != len(self.feat_dim_lab):
            raise ValueError('feat_dim_lab %s does not match data dimensions %s' %
                (repr(self.feat_dim_lab), repr(self.feat_shape)))
        if self.labels.shape[0] == 1 and self.labels.dtype == np.int:
            assert hasattr(self, 'possible_labels')
            assert isinstance(self.possible_labels, np.ndarray)
            assert len(np.setdiff1d(np.unique(self.labels), self.possible_labels)) == 0

    def check_compatibility(self, other, merge_possible_labels=False):
        # Compare features
        if (self.nfeatures != other.nfeatures):
            raise ValueError('The #features do not match (%d != %d)'
                             % (self.nfeatures, other.nfeatures))

        # Compare all other members, except data, labels and ids
        for key in self.__dict__.keys():
            if key not in other.__dict__:
                raise ValueError('Cannot add DataSets: %s not present' % key)

            v1 = self.__dict__[key]
            v2 = other.__dict__[key]

            if key not in ['_data', '_labels', '_ids', 'possible_labels', 'cl_lab']:
                if type(v1) != type(v2):
                    raise ValueError('Cannot add DataSets: %s is different' %
                            key)
                elif isinstance(v1, np.ndarray) :
                    if not np.alltrue(v1 == v2):
                        raise ValueError('Cannot add DataSets: %s is different' %
                                key)
                else:
                    if v1 != v2:
                        raise ValueError('Cannot add DataSets: %s is different' %
                                key)

            elif key == '_labels':
                if v1.dtype != v2.dtype:
                    raise ValueError('Cannot add DataSets: labels are specified differently.')

                if v1.dtype != np.int or v1.shape[0] != 1 or not merge_possible_labels:
                    if self.nclasses != other.nclasses:
                        raise ValueError('Cannot add DataSets: number of classes is different.')
                    if v1.cl_lab != v2.cl_lab:
                        raise ValueError('Cannot add DataSets: cl_lab is different.')


    def get_class(self, cl):
        """
        Construct a new DataSet containing only the instances belonging to the
        given class(es). 

        Parameters
        ----------
        cl : string, int, list of mixed strings/ints
            The class(es) to get.
            - When ``cl`` is a string, all instances of the
              class with the corresponding class label is returned.
            - When ``cl`` is an int and the DataSet labels are ints, all
              instances of the class with the corresponding value are returned.
            - When ``cl`` is an int and the DataSet labels are bools or floats,
              all instances of the class with the corresponding index will be
              returned. In this case ``cl`` must be a value between 0 and
              ``d.ninstances``.
            - When ``cl`` is a list, instances belonging to any of the
              specified classes are returned.

        Returns
        -------
        d : :class:psychic.DataSet:
            A new :class:psychic.DataSet: containing only the instances
            belonging to the specified class(es).

        See also :func:psychic.DataSet.get_nth_class:.
        """
        try:
            if isinstance(cl, list) or isinstance(cl, np.ndarray):
                idx = [self.cl_lab.index(i) if isinstance(i, str) else int(i)
                       for i in cl]
                assert np.all(idx < self.ninstances)
            elif isinstance(cl, str):
                idx = [self.cl_lab.index(cl)]
            else:
                idx = [int(cl)]
                assert idx[0] < self.ninstances
        except:
            raise ValueError('Invalid class: ' + str(cl))

        if self.labels.dtype == np.bool:
            return self[np.sum(self.labels[idx,:], axis=0).astype(np.bool)]
        else:
            idx = np.sum([self.y == i for i in idx], axis=0).astype(np.bool)
            return self[idx]

    def get_nth_class(self, n):
        ''' 
        Construct a new DataSet containing only the instances belonging to the
        given class. The desired class is given as an integer index where ``n <
        d.nclasses``. Class numbering starts at 0.

        Parameters
        ----------
        n : int
            The nth class to return. Class numbering starts at 0.

        Returns
        -------
        d : :class:psychic.DataSet:
            A new :class:psychic.DataSet: containing only the instances
            belonging to the nth class.
        '''
        if self.labels.shape[0] == 1 and self.labels.dtype == np.int:
            return self[self.y == self.possible_labels[n]]
        elif self.labels.dtype == np.bool:
            return self[self.labels[n]]
        else:
            return self[self.y == n]

    def sorted(self):
        '''A version of this DataSet sorted on the first row of :py:attr:`ids`'''
        return self[np.argsort(self.ids[0])]

    def shuffled(self):
        '''A shuffled version of this DataSet'''
        si = np.arange(self.ninstances)
        np.random.shuffle(si)
        return self[si]
        
    def __getitem__(self, i):
        if isinstance(i, slice) or isinstance(i, list) or isinstance(i, np.ndarray):
            if self.ninstances == 0:
                if not isinstance(i, slice) and len(i) == 0:
                    # Because np.zeros((0, 10)[[]] raises error, we use a workaround 
                    # using slice to index in a empty dataset.
                    # see http://projects.scipy.org/numpy/ticket/1171
                    i = slice(0) 

            return DataSet(
                data=self.data[...,i],
                labels=self.labels[:,i],
                ids=self.ids[:,i],
                default=self
            )
        elif isinstance(i, int):
            i = [i]
            return DataSet(
                data=np.atleast_2d(self.data[...,i]),
                labels=np.atleast_2d(self.labels[:,i]),
                ids=np.atleast_2d(self.ids[:,i]),
                default=self
            )
        else:
            raise ValueError, 'Unknown indexing type.'

    def __len__(self):
        '''
        Implements the :py:func:`len` function: returns number of instances.
        '''
        return self.ninstances

    def __str__(self):
        '''
        Implements the :py:func:`str` function: returns a string summary of the
        DataSet.
        '''
        return ('DataSet with %d instances, %d features [%s], %d classes: %s, '
            'extras: %s') % (self.ninstances, self.nfeatures, 
            'x'.join([str(di) for di in self.feat_shape]), 
            self.nclasses, repr(self.ninstances_per_class), repr(self.extra.keys()))

    def __repr__(self):
        return str(self)

    def __add__(a, b):
        '''Create a new DataSet by adding the instances of b to a'''
        assert(isinstance(a, DataSet))
        assert(isinstance(b, DataSet))

        # Handle empty datasets
        if a.data.size == 0:
            return b
        if b.data.size == 0:
            return a

        # Check for compatibility
        a.check_compatibility(b)

        return DataSet(
            data=np.concatenate([a.data, b.data], axis=a.data.ndim-1),
            labels=np.hstack([a.labels, b.labels]),
            ids=np.hstack([a.ids, b.ids]),
            default=a
        )

    def __eq__(a, b):
        '''
        Implements the `==` syntax: compares two DataSets.
        '''
        if not isinstance(b, DataSet):
            return False
        for member in a.__dict__.keys():
            am, bm = a.__dict__[member], b.__dict__[member]
            if isinstance(am, np.ndarray):
                if am.shape != bm.shape or not np.all(am == bm):
                    return False
            else:
                if not am == bm:
                    return False
        return True
        
    def __ne__(a, b):
        '''
        Implements the `!=` syntax: compares two DataSets.
        '''
        return not a == b
        
    @property
    def nclasses(self):
        '''
        The number of classes defined in the DataSet.
        '''
        if self.labels.shape[0] == 1 and self.labels.dtype == np.int:
            return len(self.possible_labels)
        else:
            return self.labels.shape[0]
                
    @property
    def ninstances(self):
        '''
        The number of instances defined in the DataSet.
        '''
        return self.data.shape[-1]
        
    @property
    def ninstances_per_class(self):
        '''
        A list containing for each class, the number of instances belonging to
        said class.
        '''
        if self.labels.shape[0] == 1 and self.labels.dtype == np.int:
            return [len(np.flatnonzero(self.labels[0,:] == l)) for l in self.possible_labels]
        elif self.labels.dtype == np.bool:
            return np.sum(self.labels, axis=1).astype(int).tolist()
        else:
            return np.sum(helpers.hard_max(self.labels), axis=1).astype(int).tolist()

    @property
    def prior(self):
        '''
        A list containing for each class, the fraction of instances
        belonging to said class.
        '''
        return np.asarray(self.ninstances_per_class) / float(self.ninstances)

    @property
    def feat_shape(self):
        '''
        The shape of the features defined in this DataSet.
        '''
        return self.data.shape[:-1]

    @property
    def nfeatures(self):
        '''
        The number of features defined in this DataSet.
        '''
        return reduce(operator.mul, self.feat_shape)

    def save(self, file):
        '''
        Save the DataSet to the given file. File can be specified either as a
        file object as returned by :py:func:`open` or as a string containing
        the filename.
        '''
        f = open(file, 'wb') if isinstance(file, str) else file
        cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)
        if isinstance(file, str):
            f.close()

    @classmethod
    def load(cls, file):
        '''
        Load a DataSet from the given file. File can be specified either as a
        file object as returned by :py:func:`open` or as a string containing
        the filename.
        '''
        f = open(file, 'rb') if isinstance(file, str) else file
        d = cPickle.load(f)
        assert isinstance(d, DataSet)
        d.check_consistency()
        if isinstance(file, str):
            f.close()
        return d
    
    @property
    def ix(self):
        '''
        Use this property to index the DataSet as one would do with an
        n-dimensional Numpy array, where the first dimensions are the same as
        :py:attr:`data` and the last dimension are the instances.

        For example:
        >>> # Create dataset containing 2 trials of (channels x time) data
        >>> data = [
        >>>     [[1,2,3], [4,5,6]],
        >>>     [[7,8,9], [10,11,12]],
        >>> ]
        >>> d = DataSet(data, [0,0,0], feat_dim_lab=['channels', 'time'])
        
        >>> # Select only the first channel
        >>> d.ix[0,:,:]
        
        >>> # Select only the first two samples
        >>> d.ix[:,:1,:]
        
        >>> # Select only the last instance
        >>> d.ix[:,:,-1]
        '''
        return _DataSetIndexer(self)

    @property
    def lix(self):
        '''
        Use this property to index the DataSet through the feature labels (when
        defined). The usage is similar to the :py:attr:`ix` property, but first
        a lookup is perfomed in the :py:attr:`feat_lab` attribute to determine
        the indices.

        For example:
        >>> # Create dataset containing 2 trials of (channels x time) data
        >>> data = [
        >>>     [[1,2,3], [4,5,6]],
        >>>     [[7,8,9], [10,11,12]],
        >>> ]
        >>> feat_lab = [['Fz', 'Cz'], [0.1, 0.2, 0.3]]
        >>> ids = [[0.5, 1.5]]
        >>> d = DataSet(data, [0,0,0], ids, feat_lab=feat_lab)
        
        >>> # Select only the 'Fz' channel
        >>> d.lix['Fz',:,:]
        
        >>> # Select the time range 0.1 -- 0.25, note that this contains two
        >>> # samples: 0.1 and 0.2
        >>> d.lix[:,:0.25,:]
        
        >>> # Select instance corresponding to 1.5 seconds
        >>> d.lix[:,:,1.5]
        '''
        return _DataSetLabeledIndexer(self)

    def append(self, other, ignore_index=False, merge_possible_labels=False):
        ''' Append a DataSet to this one. 
        
        Parameters
        ----------
        other : :class:`psychic.DataSet`
            The `DataSet` to append.
        ignore_index : bool (default=False)
            Ignore the ids (just number the instances).
        merge_possible_labels : bool (default=False)
            If set, the union of the possible_labels fields the two datasets
            will be taken as the new possible_labels field. This only makes
            sense if labels are specified as integers.
        '''
        assert isinstance(other, DataSet), 'Can only append other DataSets'
        return concatenate([self, other], ignore_index, merge_possible_labels)
        
class _DataSetIndexer():
    def __init__(self, d):
        self.d = d

    def __getitem__(self, key):
        if isinstance(key, tuple):
            ndim = self.d.data.ndim
            assert len(key) == ndim, ('Not enough indices (data has %d dimensions)' % ndim)

            data = self.d.data
            feat_lab = list(self.d.feat_lab)
            labels = self.d.labels
            ids = self.d.ids

            # Make selection along each axis, be careful not to drop in dimensionality
            to_drop = np.zeros(len(key), dtype=np.bool)
            for axis, ind in enumerate(key):
                if type(ind) == slice:
                    ind = np.arange(*ind.indices(data.shape[axis]))
                if type(ind) == int or type(ind) == np.int64:
                    ind = np.atleast_1d(ind)
                    to_drop[axis] = True
                else:
                    try:
                        ind = np.asarray(ind)
                    except Exception:
                        raise AssertionError('indices not convertable to a numpy array')
                
                data = data.take(ind, axis=axis)

                if axis == ndim-1:
                    # Indexing instances
                    labels = np.atleast_2d(labels[:,ind])
                    ids = np.atleast_2d(ids[:,ind])
                    to_drop[axis] = False
                    continue

                feat_lab[axis] = [feat_lab[axis][i] for i in ind]

            # Drop singleton axes
            to_drop = np.flatnonzero(to_drop)

            # Always retain at least 2 axes
            if len(to_drop) > ndim-2:
                to_drop = to_drop[1:]

            data = data.reshape([dim for axis,dim in enumerate(data.shape) if axis not in to_drop])
            feat_lab = [f for axis,f in enumerate(feat_lab) if axis not in to_drop]

            return DataSet(
                data=data,
                labels=labels,
                ids=ids,
                feat_lab=feat_lab,
                default=self.d
            )

        else:
            return self.d[key]

class _DataSetLabeledIndexer(_DataSetIndexer):
    def __init__(self, d):
        _DataSetIndexer.__init__(self, d)

    def _lookup(self, labels, index):
        try:
            return labels.index(index)
        except ValueError as e:
            if type(index) == int or type(index) == float:
                return np.array(labels).searchsorted(index)
            else:
                raise e

    def __getitem__(self, key):
        if isinstance(key, tuple):
            ndim = self.d.data.ndim
            assert len(key) == ndim, ('Not enough indices (data has %d dimensions)' % ndim)

            lab = list(self.d.feat_lab)

            # Add instance labels (DataSet.ids)
            lab += [self.d.ids.flatten().tolist()]

            new_key = []
            for axis, ind in enumerate(key):
                # Lookup indexes using lab.index()
                if type(ind) == slice:
                    if ind.start == None:
                        start = 0
                    else:
                        start = self._lookup(lab[axis], ind.start)

                    if ind.stop == None:
                        stop = self.d.data.shape[axis]
                    else:
                        stop = self._lookup(lab[axis], ind.stop)
                        if type(ind.stop) == str:
                            # Include stop point
                            stop += 1

                    new_key.append( slice(start, stop, ind.step) )

                elif type(ind) == str:
                    new_key.append( self._lookup(lab[axis], ind) )

                elif hasattr(ind, '__iter__'):
                    new_key.append( np.array([self._lookup(lab[axis], i) for i in ind]) )

                else:
                    new_key.append( self._lookup(lab[axis], ind) )

            return _DataSetIndexer.__getitem__(self, tuple(new_key))

        else:
            # Use ids for labels
            return _DataSetIndexer.__getitem__(
                self, 
                np.flatnonzero(self.ids == key)
            )

def concatenate(datasets, ignore_index=False, merge_possible_labels=False):
    """
    Efficiently concatenate multiple psychic datasets together.

    Parameters
    ----------
    datasets : list of :class:`psychic.DataSet`s
        The datasets to concatenate
    ignore_index : bool (default=False)
        If set, ignore the ``.ids`` attribute of the datasets. The resulting
        :class:`psychic.DataSet` will have an ``.ids`` attribute that simply
        numbers the instances. This can be useful when concatenating datasets
        with overlapping index labels.
    merge_possible_labels : bool (default=False)
        If set, the union of the possible_labels fields of all given datasets
        will be taken as the new possible_labels field. This only makes sense
        if labels are specified as integers.

    Returns
    -------
    d : :class:`psychic.DataSet`
        A dataset that is the result of concatenating the given datasets.
    """
    assert len(datasets) > 0
    for d in datasets:
        if not isinstance(d, DataSet):
            raise ValueError, 'Can only concatenate DataSet objects.'

    if len(datasets) == 1:
        return datasets[0]

    # Handle empty datasets
    datasets = [d for d in datasets if d.data.ndim > 0]

    # Choose the first dataset as 'base'
    base = datasets[0]

    # Check for compatibility
    for i,d in enumerate(datasets):
        # Don't compare base dataset with itself
        if i == 0:
            continue;
        base.check_compatibility(d, merge_possible_labels=True)

    # Deal with integer labels
    if merge_possible_labels and hasattr(base, 'possible_labels'):
        # Keep track of integer label -> class label mapping
        mdict = dict(reduce(operator.add, [zip(d.possible_labels, d.cl_lab) for d
            in datasets]))

        # Merge possible labels
        possible_labels = reduce(np.union1d,
            [d.possible_labels for d in datasets])

        # Make sure class labels are in the proper order
        cl_lab = [mdict[i] for i in possible_labels]
    else:
        possible_labels = None
        cl_lab = None

    # Concatenate data
    data = np.concatenate([d.data for d in datasets], axis=-1)
    labels = np.hstack([d.labels for d in datasets])

    if not ignore_index:
        ids = np.hstack([d.ids for d in datasets])
    else:
        ids = np.atleast_2d(np.arange(data.shape[-1]))

    return DataSet(data, labels, ids, possible_labels=possible_labels,
            cl_lab=cl_lab, default=base)

def as_instances(datasets, labels=None, cl_lab=None, ids=None):
    '''
    Concatenate a list of datasets into a single dataset, where the original
    datasets are instances of the new one.

    Parameters
    ----------
    datasets : list of :class:`psychic.DataSet`s
        A list of DataSets, each containing a single trial
        The feat_lab properties of the DataSets must be equal.
    labels : list or 2D-array (Default: None)
        Class labels for the trials. When not specified, each trial will be
        assigned to a separate class.
    cl_lab : list (str) (Default: None)
        Names for the classes. When not specified, nondescriptive names are
        used.
    ids : list or 2D-array (Default: None)
        For each trial, a unique label. When not specified, trials are 
        numbered.
    '''
    assert len(datasets) > 0, 'Must specify a non-empty list of DataSets'
    data_shape = datasets[0].data.shape
    for d in datasets:
        assert d.data.shape == data_shape, \
                'data.shape of all DataSets must be equal'

    data = np.concatenate([d.data[...,np.newaxis] for d in datasets], axis=-1)

    if labels != None:
        labels = np.atleast_2d(labels)
        assert labels.shape[1] == len(datasets), \
                'labels must contain a class label for each trial'
    else:
        labels = np.eye(len(datasets), dtype=np.bool)

    if cl_lab == None:
        if labels.shape[0] == 1 and labels.dtype == np.int:
            nclasses = len(np.unique(labels))
        else:
            nclasses = labels.shape[0] 

        cl_lab = ['class %02d' % (i+1) for i in range(nclasses)]

    if ids != None:
        ids = np.atleast_2d(ids)
        assert len(np.unique(ids[0,:])) == len(datasets), \
               'ids must contain an identifier for each trial'
    else:
        ids = np.atleast_2d(np.arange(len(datasets)))

    feat_lab = list(datasets[0].feat_lab) + [datasets[0].ids[0,:].tolist()]

    return DataSet(data=data, labels=labels, cl_lab=cl_lab, ids=ids,
                   feat_lab=feat_lab, default=datasets[0])

The :class:`golem.DataSet` object
=================================

The main datastructure used by Python is the :class:`golem.DataSet` class. It's
was designed for usage in machine learning so it's property names come from the
machine learning literature. This section explains how the class is used to
represent EEG data.

The following code loads an example EEG recording will produce a
:class:`golem.DataSet` object and prints it to display some summary information
about it:

>>> import psychic
>>> cont_eeg = psychic.load_bdf(psychic.find_data_path('priming-short.bdf'))
>>> print cont_eeg
DataSet with 149504 instances, 40 features [40], 1 classes: [149504], extras: []

Instances, features and classes
-------------------------------

A :class:`golem.DataSet` is a collection of *instances*, where each instance is
described by a number of *features*. An instance is a single unit that we wish to
analyze; what constitutes a 'single unit' can change along the analysis (could
be a single EEG sample, or a trial, or even a group of trials).

In the code example above, a continuous EEG recording is loaded (for example
through :func:`psychic.load_bdf`), which returns a dataset where each instance
is a single EEG sample. So there are 149504 EEG samples in the recording. Each
sample is usually recorded from many sensors at once, so the instance will have
multiple features: one for each data channel. So there are 40 data channels in
the recording.

Python's indexing and slicing syntax (``[...]``) can be used to select instances.
For example, to select the first EEG sample:

>>> print cont_eeg[0]
DataSet with 1 instances, 40 features [40], 1 classes: [1], extras: []

Each instance belongs to a certain *class*, which is a machine learning term
for group. Along the analysis, it often very useful to assign instances to
groups, so a group can be selected easily and groups can be compared. In the
above code example, each instance belongs to the same class for now.

The example recording was made from a subject that was reading either related
or unrelated word-pairs. The following code slices up the data into small
segments called trials, where each trial contains the time period during which
the subject read a single word-pair. See :ref:`trials` for more information
about trials and slicing data.

>>> event_codes = {1:'related', 2:'unrelated'}
>>> time_range = (-0.7, 1.0)
>>> trials = psychic.nodes.Slice(event_codes, time_range).train_apply(cont_eeg, cont_eeg)
>>> print trials
DataSet with 208 instances, 17400 features [40x435], 2 classes: [104, 104], extras: []

The resulting dataset has 208 instances. Each instance now corresponds to a
trial. So the subject read 208 word-pairs and each instance corresponds to the
EEG activity during the reading of each word-pair. Each instance now has 17400
features instead of two. This is because the features describing the trial are
now 40 channels x 435 samples (1.3 seconds of EEG data). The dataset defines
two classes: all trials where the subject was reading a related word-pair are
assigned to class 1 and all unrelated word-pairs are assigned to class 2. The
experiment was balanced: there were 104 related and 104 unrelated word-pairs.

The most important properties: ``ndX`` and ``Y``
------------------------------------------------

``ndX``
+++++++

The actual data (be it EEG or some result from a computation) is stored in
``ndX``: a multidimensional array. An 'array' is a term used in computer science;
mathematics calls an array with one dimension a vector, with two dimensions a
matrix and three or more dimensions a tensor. During EEG analysis, the data
can change in dimensionality.

A continuous EEG recording has two dimensions: [40 channels x 14950 samples]

>>> print cont_eeg.ndX.shape
(40, 149504)

When this recording is sliced into trials, it has three dimensions: [40
channels x 435 samples x 208 trials]

>>> print trials.ndX.shape
(40, 435, 208)

In Psychic, the first dimension is assumed to contain data channels and the second
dimension time samples. The last dimension always contains the instances:

>>> print cont_eeg.ninstances == cont_eeg.ndX.shape[1]
True
>>> print trials.ninstances == trials.ndX.shape[2]
True

``Y``
+++++

Instances are assigned to one or more classes. This mapping is stored in the
``Y`` property of the dataset. Theoretically,  ``Y`` is a matrix [classes x
instances] which contains for each instance a score indicating 'how much' it
belongs to a certain class. Practically, this means there are a few flavors of
``Y`` matrices, depending on the datatype of ``Y``:

Each instance belongs to a class yes or no
##########################################

In many cases, an instance either belongs to a class or not. In this case the
datatype of ``Y`` can be boolean. For example, to assign 6 instances to 2
classes:
    
>>> import golem
>>> import numpy as np
>>> ndX = np.zeros((4,6)) # 4 features, 6 instances
>>> Y = [[True,  True,  False, True,  False, False],
...      [False, False, True,  False, True,  True ]]
>>> print golem.DataSet(ndX=ndX, Y=Y)
DataSet with 6 instances, 4 features [4], 2 classes: [3, 3], extras: []

to assign 6 instances to 3 classes, one instance can belong to more than one
class, or to none:

>>> Y = [[True,  True,  True, False, False, False],
...      [False, True,  True, True,  False, False],
...      [False, False, True, True,  True,  False]]
>>> print golem.DataSet(ndX=ndX, Y=Y)
DataSet with 6 instances, 4 features [4], 3 classes: [3, 3, 3], extras: []

Each instance is belongs a little to each class (fuzzy assignment)
##################################################################

Instead of using boolean values, class assignment can also be done with
integers or floats. In this case, you can specify scores that indicate 'how
much' an instance belongs to a certain class. When Psychic is asked to which
class a certain instance belongs, it will reply with the class with the highest
score. Scores can be probabilities, but do not have to be (they do not have to
sum to one). For example, to assign 6 instances to 2 classes:

>>> Y = [[0.5, 0.5, 0.8, 0.9, 0.1, 0.2],
...      [0.2, 0.3, 0.6, 0.7, 0.3, 0.5]]
>>> print golem.DataSet(ndX=ndX, Y=Y)
DataSet with 6 instances, 4 features [4], 2 classes: [4, 2], extras: []

.. _informative:

Informative properties
----------------------

Apart from ``ndX`` and ``Y``, :class:`golem.DataSet` objects have many properties
to query metadata. We already saw a useful feature: printing a dataset gives a
usefull summary:

>>> print trials
DataSet with 208 instances, 17400 features [40x435], 2 classes: [104, 104], extras: []

The ``ninstances`` property is self evident:

>>> print 'There are', trials.ninstances, 'trials.'
There are 208 trials.

Which is the same as using Python's :func:`len` function:

>>> print len(trials)
208

The ``nfeatures`` property gives the number of features. Usually more informative is
the ``feat_shape`` property that gives the dimensionality of the features:

>>> print 'There are', trials.nfeatures, 'features.'
There are 17400 features.
>>> nchannels, nsamples = trials.feat_shape
>>> print 'Each trial has', nchannels, 'channels and', nsamples, 'samples.'
Each trial has 40 channels and 435 samples.
>>> print 'The shape of ndX is therefore:', trials.ndX.shape
The shape of ndX is therefore: (40, 435, 208)

With continuous EEG data, where for each instance the features are a single
vector containing the channels, the channel names can be found in ``feat_lab``:

>>> print cont_eeg.feat_lab
['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']

With trials, where the features are a matrix [channels x samples], the feature
labels can be found in ``feat_nd_lab`` instead:

>>> print trials.feat_nd_lab[0] # Labels for first dimension: channels
['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']

>>> print trials.feat_nd_lab[1][:5] # First 5 labels for second dimension: timestamps
[-0.69921875, -0.6953125, -0.69140625, -0.6875, -0.68359375]

The property ``I`` stores a unique indentifier for each instance. At the
moment, ``I`` is a matrix with only one row. This might change in the future to
a vector. To query the time stamps of the first 5 EEG samples of a continous recording:

>>> print cont_eeg.I[0, :5]
[ 0.          0.00390625  0.0078125   0.01171875  0.015625  ]

The ``I`` property of our ``trials`` dataset gives time stamps for the event onsets of
the trials:

>>> print trials.I[0, :5]
[ 32.49609375  34.99609375  37.4921875   39.9921875   42.4921875 ]

Class information can be found in ``nclasses``, ``ninstances_per_class`` and ``cl_lab``:

>>> print 'There are', trials.nclasses, 'classes'
There are 2 classes
>>> print 'Their corresponding labels are:', trials.cl_lab
Their corresponding labels are: ['related', 'unrelated']
>>> print 'The number of instances belonging to each class are:', trials.ninstances_per_class
The number of instances belonging to each class are: [104, 104]

Selecting parts of the data
---------------------------

The :class:`golem.DataSet` class supports Python's indexing and slicing syntax to select
instances and ranges of instances. Below are a few examples.

To select the first EEG sample from a continuous recording:

>>> print cont_eeg[0]
DataSet with 1 instances, 40 features [40], 1 classes: [1], extras: []

This dataset was recorded with a sample rate of 256 Hz. So to select the first second of data:

>>> print cont_eeg[:256]
DataSet with 256 instances, 40 features [40], 1 classes: [256], extras: []
>>> print 'The last time stamp:', cont_eeg[:256].I[0,-1]
The last time stamp: 0.99609375

A dataset object provides the ``ix`` property, which can be used for advanced
indexing and is therefore referred to as an *indexer*. When using the ``ix``
indexer, you can pretend to index the ``ndX`` property like you would an NumPy
array and the rest of the dataset (feature labels, class labels, etc.) will
magically follow suit:

>>> # The first two channels and all instances:
>>> print cont_eeg.ix[:2, :]
DataSet with 149504 instances, 2 features [2], 1 classes: [149504], extras: []
>>> # The first two channels and the first second of data:
>>> print cont_eeg.ix[:2, :256]
DataSet with 256 instances, 2 features [2], 1 classes: [256], extras: []
>>> # The 3rd, 4th and 20th channel, all instances (remember, indexing starts at 0):
>>> print cont_eeg.ix[[2,3,19], :]
DataSet with 149504 instances, 3 features [3], 1 classes: [149504], extras: []

This also works when ``ndX`` has more than two dimensions. For example using the
``trials`` dataset:

>>> # The first two channels:
>>> print trials.ix[:2, :, :]
DataSet with 208 instances, 870 features [2x435], 2 classes: [104, 104], extras: []
>>> # The first two channels and the first 10 trials:
>>> print trials.ix[:2, :, :10]
DataSet with 10 instances, 870 features [2x435], 2 classes: [4, 6], extras: []
>>> # The last 100 samples:
>>> print trials.ix[:, -100:, :]
DataSet with 208 instances, 4000 features [40x100], 2 classes: [104, 104], extras: []

Manually converting between samples and time and looking up indices of channels can
quickly become cumbersome. To make life easier, the dataset object also provides
the ``lix`` indexer. It works in the same manner as the ``ix`` indexer, but first
performs a lookup using the ``feat_lab``, ``feat_nd_lab`` and ``I`` properties:

>>> # Select channels 'Cz' and 'Pz', all instances:
>>> print cont_eeg.lix[['Cz', 'Pz'], :]
DataSet with 149504 instances, 2 features [2], 1 classes: [149504], extras: []
>>> # Select the first second of data, all channels:
>>> print cont_eeg.lix[:, :1]
DataSet with 256 instances, 40 features [40], 1 classes: [256], extras: []
>>> # Select time range 4 to 20 seconds for channel 'Cz':
>>> print cont_eeg.lix['Cz', 4:20]
DataSet with 4096 instances, 1 features [1], 1 classes: [4096], extras: []

And with the ``trials`` dataset:

>>> # Select channels 'Cz' and 'Pz':
>>> print trials.lix[['Cz', 'Pz'], :, :]
DataSet with 208 instances, 870 features [2x435], 2 classes: [104, 104], extras: []
>>> # Select time when first word was displayed: -0.7 to 0 seconds
>>> print trials.lix[:, -0.7:0, :]
DataSet with 208 instances, 7160 features [40x179], 2 classes: [104, 104], extras: []
>>> # Select time when second word was displayed: 0 to 1 seconds
>>> print trials.lix[:, 0:1, :]
DataSet with 208 instances, 10240 features [40x256], 2 classes: [104, 104], extras: []
>>> # Select time range leading up to the event onset (t=0):
>>> print trials.lix[:, :0, :]
DataSet with 208 instances, 7160 features [40x179], 2 classes: [104, 104], extras: []
>>> # Select all trials that were recorded in the first minute:
>>> print trials.lix[:, :, :60]
DataSet with 11 instances, 17400 features [40x435], 2 classes: [4, 7], extras: []

The ``ix`` and ``lix`` indexers can be combined, so some dimensions can be indexed
by ``ix`` and some by ``lix``, which can be very useful:

>>> # Select the first 30 trials, channels 'Cz' and 'Pz':
>>> print trials.ix[:, :, :30].lix[['Cz', 'Pz'], :, :]
DataSet with 30 instances, 870 features [2x435], 2 classes: [16, 14], extras: []

Creating new datasets
---------------------

To create a new instance of :class:`golem.DataSet`, at minumum the ``ndX`` and ``Y``
parameters should be specified:

>>> from numpy import zeros
>>> nfeatures = 4
>>> ninstances = 1000
>>> d = golem.DataSet(ndX=zeros((nfeatures, ninstances)), Y=zeros(ninstances))
>>> print d
DataSet with 1000 instances, 4 features [4], 1 classes: [1000], extras: []

In order to maintain data integrety, a dataset is read only. For example, this fails::

   d.ndX = [1,2,3]

This means that to make any changes to the data, a new dataset must be constructed. To
aid in this, the constructor of :class:`golem.DataSet` takes the parameter ``default``,
which can be set to an existing dataset. Any fields missing in the constructor will be
copied from this dataset:

>>> from numpy.random import randn
>>> d_rand = golem.DataSet(ndX=randn(nfeatures, ninstances), default=d)
>>> print d_rand
DataSet with 1000 instances, 4 features [4], 1 classes: [1000], extras: []

Any of the :ref:`informative` can be passed in the constructor as well:

>>> d_annotated = golem.DataSet(feat_lab=['feature 1', 'feature 2', 'feature 3', 'feature 4'], default=d_rand)
>>> print d_annotated.feat_lab
['feature 1', 'feature 2', 'feature 3', 'feature 4']

Loading and saving datasets
---------------------------

A dataset can be loaded with the :func:`golem.DataSet.load` function and saved
with the :func:`golem.DataSet.save` function. Both functions take a single
argument, which can either be a python file object, or a string filename::

    d = golem.DataSet.load(psychic.find_data_path('priming-trials.dat')
    d.save('some-filename.dat')

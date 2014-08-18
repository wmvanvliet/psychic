The :class:`psychic.DataSet` object
=================================

The main datastructure used by Psychic is the :class:`psychic.DataSet` class. It
is a flexible container for EEG data that supports easy mangling, chopping,
twisting and pounding of the data, while always enforcing proper meta-data,
such as channel names and time stamps. 

The main hurdle to using Psychic efficiently is the understanding of this
data structure. This is why we aim to keep its interface intuitive and adhere
to the policy: "be tolerant by accepting many different forms of input, be specific
by always producing predictable output."

Instances and features
----------------------
The following code creates a basic DataSet containing the numbers 1 to 5:

>>> import psychic
>>> d = psychic.DataSet([1,2,3,4,5])
>>> print d
DataSet with 5 instances, 1 features [1], 1 classes: [5], extras: []

During the course of your data analysis, your DataSets might hold continuous EEG data,
epoched data, spatial components, time-frequency data, etc. This is why Psychic uses
generic names for the fields of the DataSet structure, and not for example
*channels* and *time*.

A :class:`psychic.DataSet` is a collection of *instances*, where each instance is
described by a number of *features*. An instance is a single unit that we wish to
analyze; what constitutes a 'single unit' can change along the analysis. The code
above created a DataSet with 5 instances, where each instance has a single feature.

We can get at the data by using the ``data`` property:

>>> print d.data
[[1 2 3 4 5]]

Note that although we specified a 1D array in the constructor, the ``data`` property
of a DataSet always has at least two dimensions: features x instances.

This makes a DataSet that simulates 5 channels of continuous EEG data:

>>> import numpy as np
>>> d = psychic.DataSet(np.random.rand(5, 1000))
>>> print d
DataSet with 1000 instances, 5 features [5], 1 classes: [1000], extras: []

When dealing with EEG data, it's useful to add some metadata, such as channel names
and a notion of sampling rate. This information is stored in a DataSet by providing
the ``ids`` and ``feat_lab`` properties. The ``ids`` properties gives each instance a label.
Since each instance is a EEG sample, let's store timestamps as the ``ids`` property.

>>> time = np.arange(1000)/100. # 100Hz sampling rate
>>> d = psychic.DataSet(np.random.rand(5, 1000), ids=time)
>>> print d.ids
[...]

Now Psychic can deduce the sample rate for you:

>>> psychic.get_samplerate(d)
100.0

DataSets can also store feature labels in the ``feat_lab`` property. In our
example, each instance (EEG sample) has 5 features (there are 5 channels),
let's give them names:

>>> channel_names = ['AFz', 'Fz', 'Cz', 'Pz', 'Oz']
>>> time = np.arange(1000)/100. # 100Hz sampling rate
>>> d = psychic.DataSet(np.random.rand(5, 1000), ids=time, feat_lab=channel_names)
>>> print d.feat_lab
[['AFz', 'Fz', 'Cz', 'Pz', 'Oz']]

Notice that the ``feat_lab`` property is a list of lists. For each data dimension, there is
a list of feature labels. Continuous EEG data only has a single dimension for each data instance,
but, for example, epoched data has for each data instance a (channels x time) matrix as features.
In order to provide a consistent output, the ``feat_lab`` property is always a list of lists.

Class labels
------------
Each instance belongs to a certain *class*, which is a machine learning term
for group. Along the analysis, it often very useful to assign instances to
groups, so a group can be selected easily and groups can be compared. When no
class labels are provided, each instance belongs to the same class.

Psychic is very flexible in assigning class labels and supports three cases:

1. Each instance belongs to one and only one class.
2. Each instance can belong to more than one class or no class.
3. Each instance belongs to all classes to a certain degree (fuzzy assignment.) 

Integer labels
++++++++++++++
The first and quickest way to assign class labels is to simply supply a list that
maps each instance to an integer number:

>>> d = psychic.DataSet([1,2,3,4,5], labels=[1,1,2,2,1])
>>> print d
DataSet with 5 instances, 1 features [1], 2 classes: [3, 2], extras: []

In the above example, there are 2 classes (1 and 2) and each of the 5 instances
is assigned to one of them. We can now query the DataSet about classes:

>>> print d.nclasses
2
>>> print d.ninstances_per_class
[3, 2]
>>> print d.cl_lab
['class1', 'class2']

The DataSet knows how many classes there are by keeping a ``possible_labels`` property:

>>> print d.possible_labels
[1 2]

The following code produces a DataSet with 3 classes, although none of the instances
belong to the second class (maybe this is a subset of the complete data):

>>> psychic.DataSet([1,2,3,4,5], labels=[1,1,3,3,1], possible_labels=[1,2,3])
DataSet with 5 instances, 1 features [1], 3 classes: [3, 0, 2], extras: []

Descriptive names for the classes can be provided through the ``cl_lab`` property:

>>> d = psychic.DataSet([1,2,3,4,5], labels=[1,1,2,2,1], cl_lab=['target', 'distractor'])
>>> print d.cl_lab
['target', 'distractor']

When you use this style of assigning class labels, you might run into some
particularities when splitting and combining DataSets. Also, keep in mind to
keep the ``cl_lab`` and ``possible_labels`` properties aligned. The next method of assigning
class labels is a little more work to set up, but avoids these particularities.

Boolean labels
++++++++++++++
The second, and usually recommended, way to provide class labels is in the form of
a boolean matrix, where each rows corresponds to a class and each column assigns
a data instance to said class. For example:

>>> d = psychic.DataSet([1,2,3,4,5], labels=[[True, True, False, False, False],
...                                          [False, False, True, True, True]])
>>> print d
DataSet with 5 instances, 1 features [1], 2 classes: [2, 3], extras: []

Now an instance can be assigned to multiple classes at the same time:

>>> d = psychic.DataSet([1,2,3,4,5], labels=[[True, True, False, False, False],
...                                          [True, True, True, True, True]])
>>> print d
DataSet with 5 instances, 1 features [1], 2 classes: [2, 5], extras: []

Also, classes can straightforwardly exists with no instances assigned to them:

>>> d = psychic.DataSet([1,2,3,4,5], labels=[[True, True, False, False, False],
...                                          [False, False, True, True, True],
...                                          [False, False, False, False, False]])
>>> print d
DataSet with 5 instances, 1 features [1], 3 classes: [2, 3, 0], extras: []

Float labels
++++++++++++
The final method of assigning class labels is using a matrix of floats. In this manner,
each instance is assigned to each class with a certain score (which may be a
probability, if you keep the scores normalized). When printing DataSets, the output
algorithm will assign each instance to the class with the highest score:

>>> d = psychic.DataSet([1,2,3,4,5], labels=[[1.0, 0.7, 0.1, 0.2, 0.0],
...                                          [0.0, 0.1, 0.5, 0.7, 0.1],
...                                          [0.0, 0.2, 0.4, 0.1, 0.9]])
>>> print d
DataSet with 5 instances, 1 features [1], 3 classes: [2, 2, 1], extras: []


Modifying DataSets and copying data from other DataSets 
-------------------------------------------------------
Once a DataSet has been created, its properties cannot be altered. It must be this
way to ensure that data integrety is conserved along the way. For example, Psychic must
ensure that the number of feature labels always equals the number of features.

To avoid having to specify all the properties every time a mutation on a DataSet is 
desired, the constructor takes the ``default`` argument to specify a DataSet to use
for taking default values for all properties. When specified, all properties
that are not given in the constructor, but are present in the default DataSet
and are compatible, are transferred to the new one.

>>> d = psychic.DataSet([1,2,3,4,5], feat_lab=['Fz'], ids=[0.1, 0.2, 0.3, 0.4, 0.5])
>>> d2 = psychic.DataSet([[1,2,3,4,5],
...                       [6,7,8,9,10]], default=d)
>>> print d2.ids
[[ 0.1  0.2  0.3  0.4  0.5]]
>>> print d2.feat_lab
[[0, 1]]

In the example above, the ``ids`` property was compatible and was transferred,
but the ``feat_lab`` property was incompatible, so instead some default feature
labels were generated.

A common way of modifying DataSet is to specify the original DataSet in the
``default`` argument and only supply the properties that actually changed:

>>> d = psychic.DataSet([1,2,3,4,5], labels=[1,1,2,2,2],
...     cl_lab=['target', 'distractor'], feat_lab=['Fz'],
...     ids=[0.1, 0.2, 0.3, 0.4, 0.5])
>>> d2 = psychic.DataSet([[1,2,3,4,5],
...                       [6,7,8,9,10]], feat_lab=['Fz', 'Cz'], default=d)
>>> print d2
DataSet with 5 instances, 2 features [2], 2 classes: [2, 3], extras: []

Selecting parts of the data
---------------------------

The :class:`psychic.DataSet` class supports Python's indexing and slicing syntax to select
instances and ranges of instances. Below are a few examples.

To select the first EEG sample from a continuous recording:

>>> # Load recording from BDF file
>>> cont_eeg = psychic.load_bdf(psychic.find_data_path('priming-short.bdf'))
>>> print cont_eeg
DataSet with 149504 instances, 40 features [40], 3 classes: [148792, 355, 357], extras: []

>>> # Select first sample
>>> print cont_eeg[0]
DataSet with 1 instances, 40 features [40], 3 classes: [1, 0, 0], extras: []

This dataset was recorded with a sample rate of 256 Hz. So to select the first second of data:

>>> print cont_eeg[:256]
DataSet with 256 instances, 40 features [40], 3 classes: [256, 0, 0], extras: []
>>> print 'The last time stamp:', cont_eeg[:256].ids[0,-1]
The last time stamp: 0.99609375

A dataset object provides the ``ix`` property, which can be used for advanced
indexing and is therefore referred to as an *indexer*. When using the ``ix``
indexer, you can pretend to index the ``data`` property like you would an NumPy
array and the rest of the dataset (feature labels, class labels, etc.) will
magically follow suit:

>>> # The first two channels and all instances:
>>> print cont_eeg.ix[:2, :]
DataSet with 149504 instances, 2 features [2], 3 classes: [148792, 355, 357], extras: []
>>> # The first two channels and the first second of data:
>>> print cont_eeg.ix[:2, :256]
DataSet with 256 instances, 2 features [2], 3 classes: [256, 0, 0], extras: []
>>> # The 3rd, 4th and 20th channel, all instances (remember, indexing starts at 0):
>>> print cont_eeg.ix[[2,3,19], :]
DataSet with 149504 instances, 3 features [3], 3 classes: [148792, 355, 357], extras: []

This also works when ``data`` has more than two dimensions. For example using the
``trials`` dataset:

>>> trials = psychic.DataSet.load(psychic.find_data_path('priming-trials.dat'))
>>> # The first two channels:
>>> print trials.ix[:2, :, :]
DataSet with 800 instances, 1382 features [2x691], 2 classes: [400, 400], extras: []
>>> # The first two channels and the first 10 trials:
>>> print trials.ix[:2, :, :10]
DataSet with 10 instances, 1382 features [2x691], 2 classes: [1, 9], extras: []
>>> # The last 100 samples:
>>> print trials.ix[:, -100:, :]
DataSet with 800 instances, 3500 features [35x100], 2 classes: [400, 400], extras: []

Manually converting between samples and time and looking up indices of channels can
quickly become cumbersome. To make life easier, the dataset object also provides
the ``lix`` indexer. It works in the same manner as the ``ix`` indexer, but first
performs a lookup using the ``feat_lab``, ``feat_nd_lab`` and ``I`` properties:

>>> # Select channels 'Cz' and 'Pz', all instances:
>>> print cont_eeg.lix[['Cz', 'Pz'], :]
DataSet with 149504 instances, 2 features [2], 3 classes: [148792, 355, 357], extras: []
>>> # Select the first second of data, all channels:
>>> print cont_eeg.lix[:, :1]
DataSet with 256 instances, 40 features [40], 3 classes: [256, 0, 0], extras: []
>>> # Select time range 4 to 20 seconds for channel 'Cz':
>>> print cont_eeg.lix['Cz', 4:20]
DataSet with 4096 instances, 1 features [1], 3 classes: [4096, 0, 0], extras: []

And with the ``trials`` dataset:

>>> # Select channels 'Cz' and 'Pz':
>>> print trials.lix[['Cz', 'Pz'], :, :]
DataSet with 800 instances, 1382 features [2x691], 2 classes: [400, 400], extras: []
>>> # Select time when first word was displayed: -0.7 to 0 seconds
>>> print trials.lix[:, -0.7:0, :]
DataSet with 800 instances, 6265 features [35x179], 2 classes: [400, 400], extras: []
>>> # Select time when second word was displayed: 0 to 1 seconds
>>> print trials.lix[:, 0:1, :]
DataSet with 800 instances, 8960 features [35x256], 2 classes: [400, 400], extras: []
>>> # Select time range leading up to the event onset (t=0):
>>> print trials.lix[:, :0, :]
DataSet with 800 instances, 6265 features [35x179], 2 classes: [400, 400], extras: []
>>> # Select all trials that were recorded in the first minute:
>>> print trials.lix[:, :, :60]
DataSet with 0 instances, 24185 features [35x691], 2 classes: [0, 0], extras: []

The ``ix`` and ``lix`` indexers can be combined, so some dimensions can be indexed
by ``ix`` and some by ``lix``, which can be very useful:

>>> # Select the first 30 trials, channels 'Cz' and 'Pz':
>>> print trials.ix[:, :, :30].lix[['Cz', 'Pz'], :, :]
DataSet with 30 instances, 1382 features [2x691], 2 classes: [16, 14], extras: []


Loading and saving datasets
---------------------------

A dataset can be loaded with the :func:`psychic.DataSet.load` function and saved
with the :func:`psychic.DataSet.save` function. Both functions take a single
argument, which can either be a python file object, or a string filename:

>>> d = psychic.DataSet.load(psychic.find_data_path('priming-trials.dat'))
>>> d.save('some-filename.dat')

See also :doc:`dataformats` to load from and save to other formats.

 

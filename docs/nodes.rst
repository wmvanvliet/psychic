Nodes
=====

Most of the functionality in Psychic is provided in the form of 'Nodes'. 
See :ref:`the API documentation on nodes <api-nodes>` for an overview of the multitude of nodes available in Psychic.

A node is a class which has two main methods:

.. function:: Node.train(d)

    The term 'train' comes from the machine-learning literature as Psychic is foremost
    envisioned as a framework for brain-computer interfacing, of which machine-learning is an
    integral part.

    This function examines the data *d*, which is supplied as a :class:`psychic.DataSet`. It
    stores any parameters it needs internally. This function returns the trained node. Not all
    nodes require training, but most do.

.. function:: Node.apply(d)

    Performs the functionality of node on the data *d*, which is supplied as a
    :class:`psychic.DataSet`. Returns the resulting data.

And a convenience method:

.. function:: Node.train_apply(train, test)

    Convenience function that calls both the ``train`` and the ``apply``
    methods in sequence. If the ``test`` argument is omitted, both training and
    application of the node is performed on the data specified in the ``train``
    argument.

For example, to create a node that downsamples the signal to 100 Hz:

>>> import psychic
>>> downsample = psychic.nodes.Resample(100.0)
>>> print downsample
Resample

Before we can use the node, it must know what the current samplerate of the
signal is. Therefore it must be 'trained' on the data, before it can be 'applied'.

Creating some fake data (10 Hz sine waves, sampled at 2000 Hz):

>>> eeg = psychic.fake.sine(freq=10, nchannels=4, duration=5, sample_rate=2000)

Training the node:

>>> print downsample.train(eeg)
Resample

The learned parameters can now be queried:

>>> print downsample.old_samplerate, downsample.new_samplerate
2000.0 100.0

To do the actual downsampling:

>>> downsampled = downsample.apply(eeg)
>>> print psychic.get_samplerate(downsampled)
100.0

A convenience function `train_apply` exists to do both steps in one command:

>>> downsampled = downsample.train_apply(eeg)

Chaining nodes
--------------

To quickly apply multiple operations on the data, nodes can be chained, using
the special :class:`psychic.nodes.Chain` node. A chain of nodes behaves like a
single node that performs the intermediate steps in sequence.

For example, to first band-pass filter the signal and then downsample:

>>> filter = psychic.nodes.Butterworth(4, [0.5, 30])
>>> downsample = psychic.nodes.Resample(100.0)
>>> chain = psychic.nodes.Chain([filter, downsample])
>>> filtered_downsampled = chain.train_apply(eeg)

Chains can contain an entire BCI pipeline, for example a SSVEP classifier:

>>> ssvep_data = psychic.DataSet.load(psychic.find_data_path('ssvep-cluedo.dat'))
>>> pipeline = psychic.nodes.Chain([
...     psychic.nodes.Butterworth(2, [1, 30]),
...     psychic.nodes.SlidingWindow(win_size=20, win_step=20),
...     psychic.nodes.CanonCorr(sample_rate=128, frequencies=[60/7.,  60/6., 60/5., 60/4.])
... ])
>>> print psychic.perf.accuracy(pipeline.train_apply(ssvep_data))
1.0


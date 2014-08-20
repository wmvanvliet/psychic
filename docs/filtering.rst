Filtering
=========

Psychic offers a variety of filters, both operating on the :ref:`time domain` (for
example, a :ref:`frequency filter` or a :ref:`moving average`) and on the :ref:`sensor domain`
(for example :ref:`laplacian filtering` or :ref:`common spatial patterns`).

.. _time domain:

Time domain filters
-------------------

A time domain filter operates along the time dimension. Tasks for these filters include
frequency filtering and downsampling of the signal.

.. _frequency filter:

Frequency filters
+++++++++++++++++

To apply a band-pass, band-stop, low-pass or high-pass frequency filter, the
:class:`psychic.nodes.Butterworth` node can be used. This node is a wrapper around
SciPy's :func:`scipy.signal.iirfilter` and :func:`scipy.signal.filtfilt` functions.

As an alternative, the :class:`psychic.nodes.FFTFilter` node can be used. It uses
the (inverse) fourier transform to filter the data. With a large dataset, this can
be slow and memory intensive, but it generally generates a sharper cutoff than
an IIR filter.

Some commonly used filters:
###########################

Generating a sample EEG signal to work on:

>>> import psychic
>>> d = psychic.fake.gaussian(nchannels=4, duration=10, sample_rate=256)

A 4th order Butterworth IIR low-pass filter, suppressing all frequencies **above** 30 Hz:

>>> filter = psychic.nodes.Butterworth(4, 0.5, btype='lowpass')

A 4th order Butterworth IIR high-pass filter, suppressing all frequencies **below** 0.5 Hz:

>>> filter = psychic.nodes.Butterworth(4, 30, btype='highpass')

A 4th order Butterworth IIR band-pass filter between 0.5-30 Hz:

>>> filter = psychic.nodes.Butterworth(4, [0.5, 30], btype='bandpass')

Applying the band-pass filter:

>>> d_filt = filter.train_apply(d, d)

Creating and applying an FFT filter between 0.5-30 Hz:

>>> filter = psychic.nodes.FFTFilter(0.5, 30)
>>> d_filt = filter.train_apply(d, d)

Downsampling the signal
+++++++++++++++++++++++

EEG amplifiers can digitize the signal at exceedingly high rates (thousands of Hz). After
low-pass filtering such a high sample rate is usually not required. To downsample the signal
the :class:`psychic.nodes.Resample` node can be used. During training, the node will estimate
the current sample rate using the :func:`psychic.get_samplerate` function. When applied, the
signal is downsampled to the requested sample rate.

Simulating EEG signals sampled at 2000 Hz:

>>> import psychic
>>> d = psychic.fake.sine(freq=10, nchannels=4, duration=1.0, sample_rate=2000)
>>> print psychic.get_samplerate(d)
2000.0

Resampling the signal to 100 Hz:

>>> resample = psychic.nodes.Resample(100)
>>> d_resampled = resample.train_apply(d, d)
>>> print psychic.get_samplerate(d_resampled)
100.0

.. _sensor domain:

Sensor domain filters
---------------------

A sensor domain filters operate long the EEG channels. Usually these are so
called linear spatial filters: the ouput is a linear mixture of the available channels:

.. math::
    \newcommand{mat}[1]{\mathrm{\bf #1}}
    \mat{X}' = \mat{W}^\intercal \cdot \mat{X}

Where :math:`\mat{X}` is the [channels x samples] EEG signal and :math:`\mat{X}'` is a
[components x samples] matrix containing the result. Matrix :math:`\mat{W}` contains in each column a
spatial filter, that combines the original channels into one 'channel' that is
called a component from now on. The dimensions of :math:`\mat{W}` are therefore
(channels x components).

For example, a spatial filter :math:`\mat{W}` that creates two components:

1. the average of the channels 1-4
2. the average of channels 5-8

would look like this:

>>> import numpy as np
>>> W = np.array([[ 0.25,  0.  ],
...               [ 0.25,  0.  ],
...               [ 0.25,  0.  ],
...               [ 0.25,  0.  ],
...               [ 0.  ,  0.25],
...               [ 0.  ,  0.25],
...               [ 0.  ,  0.25],
...               [ 0.  ,  0.25]])

To create a new spatial filter, the :class:`psychic.nodes.SpatialFilter` can be used.

>>> filter = psychic.nodes.SpatialFilter(W)
>>> d = psychic.DataSet(np.random.randn(8, 100))
>>> print filter.apply(d)
DataSet with 100 instances, 2 features [2], 1 classes: [100], extras: []

Psychic comes with a multitude of spatial filters. See the :ref:`API documentation on spatial filters <api-spat>`.

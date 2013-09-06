Managing trials
===============

A trial is a segment of EEG data related to a specific event. These events could
be the presentation of a stimulus on the screen, the onset of a button press,
etc. When analyzing such EEG recordings, its useful to cut out segments around
the onset of the events. Later, these segments can for example be averaged to
study event-related potentials, or these segments could be used as input for a
classifier to construct a brain-computer interface.

Slicing
-------

To extract trials from a continuous recording, the :class:`psychic.nodes.Slice`
node can be used. It assumes that the recording provides a marker stream: a
channel which contains numerical codes indicating the type of event occuring.
The signal starts at 0. At the onset of an event, the signal jumps to a value
which corresponds to the type of event and remains there for the duration of the
event. After the event, the signal either returns to 0 or jumps to a different
value, indicating the onset of a different event.

.. figure::  images/marker_stream.png
   :align:   center

   Graphical representation of an example marker stream

Example generating continuous EEG recording with a marker stream:

>>> import psychic
>>> import golem
>>> import numpy as np
>>> d = psychic.fake.gaussian(nchannels=4, duration=10, sample_rate=100)
>>> # Y contains the marker stream
>>> Y = np.zeros(d.ninstances)
>>> Y[5::100] = 1
>>> Y[55::100] = 2
>>> d_marked = golem.DataSet(Y=Y, default=d)
>>> print d_marked
DataSet with 1000 instances, 4 features [4], 1 classes: [1000], extras: []

To extract trials by slicing segments starting 200 ms before the onset of the
event lasting to 500 ms after:

>>> mdict = {1:'event type 1', 2:'event_type 2'}
>>> slicer = psychic.nodes.Slice(mdict, (-0.2, 0.5))
>>> trials = slicer.train_apply(d_marked, d_marked)
>>> print trials
DataSet with 18 instances, 280 features [4x70], 2 classes: [9, 9], extras: []

Using a sliding window
----------------------

Another commonly used approach to obtain trials to use apply a sliding window.

.. figure::  images/sliding_window.png
   :align:   center

   Sliding window with size 2 and step 1.3 (seconds). 

This operation can be performed with the :class:`psychic.nodes.SlidingWindow`
node. The example below extracts trials using a sliding window of 2 seconds that
is moved across the signal in steps of 1.3 seconds:

>>> import psychic
>>> d = psychic.fake.gaussian(nchannels=4, duration=10, sample_rate=100)
>>> window = psychic.nodes.SlidingWindow(win_size=2, win_step=1.3)
>>> trials = window.train_apply(d, d)
>>> print trials
DataSet with 7 instances, 800 features [4x200], 1 classes: [7], extras: []

Baselining trials
-----------------

Due to drifts in the EEG signal, the mean signal of a trial is not always
aligned to zero. Especially during ERP analysis, it's important to align the
trials to some meaningful 'baseline' voltage in order to compare them. The
:class:`psychic.nodes.Baseline` node calculates a baseline and aligns the signal
accordingly.

>>> mdict = {1:'event type 1', 2:'event_type 2'}
>>> slicer = psychic.nodes.Slice(mdict, (-0.2, 0.5))
>>> trials = slicer.train_apply(d_marked, d_marked)
>>> baseliner = psychic.nodes.Baseline((-0.2, 0))
>>> trials_baselined = baseliner.train_apply(trials, trials)

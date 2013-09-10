Plotting functions
==================

Psychic has a variety of plotting functions that are useful to quickly
visualize the EEG data. They also serve as a good basis for publication-ready
plots.

Plotting raw EEG
----------------

The :func:`psychic.plot_eeg` function expects a continuous EEG recoding and
will plot it. The channels are shown stacked on top of each other with a bit
of spacing, which can be controlled with the ``vspace`` parameter. If this
parameter is omitted, Psychic will attempt to estimate a sane value for it.
If the data contains event markers, the onsets of those will be shown as
vertical lines.

.. warning::
    This function plots **all** data that it is given. Don't try to plot an
    hour long EEG recording in it's entirety. Use the ``.lix`` property of
    :class:`golem.DataSet` to select a piece of data first.

For example:

>>> import psychic
>>> d = psychic.fake.gaussian(nchannels=4, nsamples=1000, sample_rate=100)
>>> psychic.plot_eeg(d, vspace=3)

Will produce:

.. figure::  images/plot_eeg.png
   :align:   center

   Plotting some random data with :func:`psychic.plot_eeg`.

Zooming in on the two seconds alone:

>>> psychic.plot_eeg(d.lix[:, 2:4], vspace=5)

.. figure::  images/plot_eeg_zoom.png
   :align:   center

   Two seconds of EEG data.

Notice that the time axis keeps track of absolute time. In the above example,
we cut the data from 2 to 4 seconds and the time axis reflects that. To make
the time axis start at zero, the ``start`` parameter can be supplied, which 
marks which time to take as t=0:

>>> psychic.plot_egg(d.lix[:, 2:4], vspace=5, start=2)

.. figure::  images/plot_eeg_zoom_start.png
   :align:   center

   First two seconds of EEG data.


Plotting ERPs
-------------

>>> import psychic
>>> import golem
>>> trials = golem.DataSet.load(psychic.find_data_path('priming-trials.dat'))
>>> psychic.plot_erp(trials.lix[['Fz', 'Cz', 'Pz', :, :])

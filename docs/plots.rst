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
    :class:`psychic.DataSet` to select a piece of data first.

For example:

>>> import psychic
>>> d = psychic.fake.gaussian(nchannels=4, duration=10, sample_rate=100)
>>> fig = psychic.plot_eeg(d, vspace=3)

Will produce:

.. figure::  images/plot_eeg.png
   :align:   center

   Plotting some random data with :func:`psychic.plot_eeg`.

Zooming in on two seconds:

>>> fig = psychic.plot_eeg(d.lix[:, 2:4], vspace=5)

.. figure::  images/plot_eeg_zoom.png
   :align:   center

   Two seconds of EEG data.

Notice that the time axis keeps track of absolute time. In the above example,
we cut the data from 2 to 4 seconds and the time axis reflects that. To make
the time axis start at zero, the ``start`` parameter can be supplied, which 
marks which time to take as t=0:

>>> fig = psychic.plot_eeg(d.lix[:, 2:4], vspace=5, start=2)

.. figure::  images/plot_eeg_zoom_start.png
   :align:   center

   First two seconds of EEG data, how with timestamps starting at 0.

Plotting ERPs
-------------

When data has been split into trials (see :ref:`trials`), a so called
Event-Related Potential plot can be constructed. For each class, the
corresponding trials are averaged to construct the ERP. ERPs for the different
classes are plotted on top of each other in order to compare them.

The :func:`psychic.plot_erp` function bundles a lot of functionality into one
function. The function tries by default to obtain a lot of relevant information
about the ERPs with a simple command to quickly get an overview. Next, the user
can tweak the many parameters to show only the desired information in the
desired manner.

Basic usage:

>>> import psychic
>>> trials = psychic.DataSet.load(psychic.find_data_path('priming-trials.dat'))
>>> trials = psychic.nodes.Baseline((-0.2, 0)).train_apply(trials, trials)
>>> fig = psychic.plot_erp(trials.lix[['Fz', 'Cz', 'Pz'], :, :]);

.. figure::  images/plot_erp.png
    :align:   center

    ERP of two classes. Channels 'Fz', 'Cz' and 'Pz' are selected. Shaded areas
    indicate the outcome of sample by sample t-tests (p < 0.05, not corrected for
    multiple comparisons).

For large datasets with many trials it might be inefficient to recalculate the
ERP every time. As an alternative to feeding all trials into
:func:`psychic.plot_erp`, ERPs can be calculated beforehand using
:class:`psychic.nodes.ERP`. However, this prevents :func:`psychic.plot_erp` to
calculate statistic tests and displaying the amount of trials used to create
the ERPs.

>>> erp = psychic.nodes.ERP().train_apply(trials, trials)
>>> fig = psychic.plot_erp(erp.lix[['Fz', 'Cz', 'Pz'], :, :])

.. figure::  images/plot_erp_erp.png
    :align:   center

    ERP of two classes. Channels 'Fz', 'Cz' and 'Pz' are selected.

API Reference
=============

This is the full API documentation. Use this to look up specific classes and
functions.

Nodes
-----

.. currentmodule:: psychic

All the nodes provided by Psychic:

Filtering
+++++++++

.. autosummary::
    :toctree: generated/
    :template: node.rst

    nodes.Butterworth
    nodes.FFTFilter
    nodes.Filter
    nodes.Resample
    nodes.Decimate

Working with trials
+++++++++++++++++++

.. autosummary::
    :toctree: generated/
    :template: node.rst

    nodes.Slice
    nodes.SlidingWindow
    nodes.Baseline

SSVEP
+++++

Classifiers for Steady State Visual Evoked Potentials

.. autosummary::
    :toctree: generated/
    :template: node.rst

    nodes.MNEC
    nodes.SLIC
    nodes.CanonCorr

Functions
---------

File formats
++++++++++++

.. autosummary::
    :toctree: generated/
    :template: function.rst

    load_edf
    load_bdf

Working with trials
+++++++++++++++++++
.. autosummary::
    :toctree: generated/
    :template: function.rst

    slice
    concatenate_trials
    baseline
    erp
    ttest

Utility functions
+++++++++++++++++

.. autosummary::
    :toctree: generated/
    :template: function.rst

    get_samplerate

Simulating data
+++++++++++++++

.. autosummary::
    :toctree: generated/
    :template: function.rst

    fake.sine
    fake.gaussian
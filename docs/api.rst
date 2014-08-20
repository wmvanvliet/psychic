API Reference
=============

This is the full API documentation. Use this to look up specific classes and
functions.

Data sets
---------

Classes and functions concerning data sets

.. currentmodule:: psychic

.. autosummary::
    :toctree: generated/
    :template: class.rst

    DataSet

.. autosummary::
    :toctree: generated/
    :template: function.rst

    concatenate
    as_instances

.. _api-nodes:

Nodes
-----

All the nodes provided by Psychic:

Base Nodes
++++++++++
.. autosummary::
    :toctree: generated/
    :template: node.rst

    nodes.Chain
    nodes.ApplyOverFeats
    nodes.ApplyOverInstances

Referencing
+++++++++++
.. autosummary::
    :toctree: generated/
    :template: node.rst

    nodes.EEGMontage

Temporal filtering
++++++++++++++++++

.. autosummary::
    :toctree: generated/
    :template: node.rst

    nodes.Filter
    nodes.OnlineFilter
    nodes.Butterworth
    nodes.FFTFilter
    nodes.Resample
    nodes.Decimate
    nodes.Winsorize

Time-frequency decomposition
++++++++++++++++++++++++++++
.. autosummary::
    :toctree: generated/
    :template: node.rst

    nodes.TFC

.. _api-spat:

Non-learning spatial filters
++++++++++++++++++++++++++++
.. autosummary::
    :toctree: generated/
    :template: node.rst

    nodes.SpatialFilter
    nodes.CAR
    nodes.Whiten
    nodes.SymWhitening
    nodes.SlowSphering
    nodes.SpatialBlur
    nodes.AlignedSpatialBlur

Learning spatial filters
++++++++++++++++++++++++
.. autosummary::
    :toctree: generated/
    :template: node.rst

    nodes.CSP
    nodes.SpatialSNR
    nodes.SpatialFC
    nodes.SpatialCFMS
    nodes.Deflate

Working with trials
+++++++++++++++++++
.. autosummary::
    :toctree: generated/
    :template: node.rst

    nodes.Slice
    nodes.SlidingWindow
    nodes.Baseline
    nodes.ERP

Steady-State Visual Evoked Potentals
++++++++++++++++++++++++++++++++++++
.. autosummary::
    :toctree: generated/
    :template: node.rst

    nodes.SLIC
    nodes.SSVEPNoiseReduce
    nodes.MNEC
    nodes.CanonCorr
    nodes.MSI

EOG artifact correction
+++++++++++++++++++++++
.. autosummary::
    :toctree: generated/
    :template: node.rst

    nodes.EOGCorr

Baseline classifiers
++++++++++++++++++++

Use :ref:`scikit-learn <scikit-learn>` for a good collection of machine learning
algorithms. Psychic only provides some baseline classifiers to use for sanity
checking and debug purposes.
    
.. autosummary::
    :toctree: generated/
    :template: node.rst

    nodes.PriorClassifier
    nodes.RandomClassifier
    nodes.WeakClassifier

Beamformers
+++++++++++
.. autosummary::
    :toctree: generated/
    :template: node.rst

    nodes.TemplateFilter
    nodes.GaussTemplateFilter

File formats
------------

BDF
+++
.. autosummary::
    :toctree: generated/
    :template: class.rst

    BDFReader
    BDFWriter

.. autosummary::
    :toctree: generated/
    :template: function.rst

    load_bdf
    save_bdf

EDF/EDF+
++++++++
.. autosummary::
    :toctree: generated/
    :template: function.rst

    load_edf

Functions
---------


Working with trials
+++++++++++++++++++
.. autosummary::
    :toctree: generated/
    :template: function.rst

    slice
    baseline
    erp
    ttest

Utility functions
+++++++++++++++++

.. autosummary::
    :toctree: generated/
    :template: function.rst

    find_data_path
    get_samplerate
    helpers.to_one_of_n

Simulating data
+++++++++++++++

.. autosummary::
    :toctree: generated/
    :template: function.rst

    fake.gaussian
    fake.sine

Plotting
++++++++

.. autosummary::
    :toctree: generated/
    :template: function.rst

    plot_eeg
    plot_erp
    plot_erp_image
    plot_erp_psd
    plot_erp_specgrams
    plot_psd
    plot_scalpgrid
    plot_specgrams

Cross-validation
++++++++++++++++
.. autosummary::
    :toctree: generated/
    :template: function.rst

    cv.cross_validate
    cv.cross_validation_sets
    cv.rep_cv
    cv.seq_splits
    cv.strat_splits
    
Statistics
++++++++++
.. autosummary::
    :toctree: generated/
    :template: function.rst

    stat.auc_confidence
    stat.benjamini_hochberg
    stat.bonferroni
    stat.bonferroni_holm
    stat.mut_inf

Classifier performance metrics
++++++++++++++++++++++++++++++
.. autosummary::
    :toctree: generated/
    :template: function.rst

    perf.accuracy
    perf.auc
    perf.class_loss
    perf.conf_mat
    perf.format_confmat
    perf.mean_std
    perf.mutinf

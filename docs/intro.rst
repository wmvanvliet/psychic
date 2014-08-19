Introducing Psychic
===================

Psychic is a `Python <http://www.python.org>`_ package aimed to reduce the
workload of the EEG researcher by offering a datastructure that facilitates
chopping/rearranging/crunching/mangling of EEG data, as well as a robust
implementation of common operations. Psychic assumes that you are doing
original research and want to do stuff that is not implemented, so it tries to
keep out of your way and avoid tedious bookkeeping of metadata as much as
possible, leaving you free to focus on getting your algorithm working.

Psychic is written by two PhD's that constructed it during their thesises.
The basic workflow for a new algorithm looks like this:

1. Load data
2. Do standard processing stuff, like re-referencing, band-pass filtering, etc.
3. Start ugly Python script that tries to do something new with the data that
   nobody has ever tried before.
4. Bang on the Python script until it somewhat works.
5. Write slightly less ugly Python script that works better.
6. Write paper about your new results.
7. Modify algorithm to be re-usable, write documentation and unit tests, make
   it part of Psychic.
8. Share implementation of the algorithm with your collegues and easily use it
   in your next project.

Psychic aims to be helpful at every step of the way, not just at steps 1,2 and 8.

Psychic at a glance
-------------------

>>> import psychic

>>> # Load data
>>> d = psychic.load_bdf(psychic.find_data_path() + 'priming.bdf')
 
>>> # Description of the EEG montage.
>>> # Reference electrodes were placed on the mastoids, using the unhelpful
>>> # channel names EXG1, EXG2.
>>> montage = psychic.nodes.EEGMontage(ref=['EXG1', 'EXG2'])
 
>>> # Re-reference the signal
>>> d = montage.train_apply(d)
 
>>> # Band-pass filter, 4th order Butterworth
>>> d = psychic.nodes.Butterworth(4, [0.3, 30]).train_apply(d)
 
>>> # Cut trials from -0.2 to 1.0 seconds relative to the stimulus onset
>>> marker_code_to_class = {1:'related', 2:'unrelated'}
>>> trials = psychic.nodes.Slice(marker_code_to_class, [-0.2, 1.0]).train_apply(d)

>>> # Plot ERPs
>>> psychic.plot_erp(trials)
 
>>> # Select only channels Pz, P3 and P4, from 0.3 to 0.6 seconds
>>> trials = trials.lix[['Pz', 'P3', 'P4'], 0.3:0.6, :]
 
>>> # Downsample to 100 Hz
>>> trials = psychic.nodes.Resample(100).train_apply(trials)
 
>>> # 50/50 split in train and test set
>>> ntrials = len(trials)
>>> train, test = trials[:ntrials/2], trials[ntrials/2:]
 
>>> # No wait, ensure equal number of trials per class when doing the split
>>> train, test = psychic.cv.strat_splits(trials, 2)
 
>>> # Run a classifier from scikit-learn on the data
>>> from sklearn.svm import LinearSVC
>>> cl = LinearSVC().fit(train.X, train.y)
>>> print cl.score(test.X, test.y)
0.6025

Dive in
-------

Follow along with the :doc:`index` that will take you through all the
major features of Psychic. Hack away with the `BCI tutorials <https://sites.google.com/site/wmvanvliet/tutorials/eeg-signal-analysis>`_.
For the fine details of every class and function, see the :doc:`api`.

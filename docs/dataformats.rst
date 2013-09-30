Loading from and saving to common dataformats
=============================================

Currently, Psychic provides support for loading EDF and BDF files:

EDF and EDF+ files
    The European Data Format is used by many EEG amplifiers. The EDF+ format is a
    more modern version of this format. Use :func:`psychic.load_edf` to load it.

BDF files
    The BioSemi Data Format is used by the BioSemi ActiveTwo EEG amplifier. It is
    a modified version of the EDF format. Use :func:`psychic.load_bdf` to load it.

Both functions will return a :class:`golem.DataSet` with the following fields:

 - ``d.X`` the [channels x samples] EEG data
 - ``d.Y`` the status channel
 - ``d.feat_lab`` the channel names
 - ``d.I`` timestamps for each sample

Re-referencing
--------------

EEG measures voltage differences between each of the electrodes and some common
reference.  The choise of reference can greatly influence your signal-to-noise
ratio (SNR) and the shape the of the EEG in general. A commonly used reference
is 'linked mastoids', where an electrodes are placed behind each ear and the
average signal of the two electrodes is taken as reference.  It is likely that
the EEG recorder stores the data references to some default location. For
example, the BioSemi stores it's data referenced to the CMS electrode. When this
is the case, you need to re-reference the signal.

The :func:`psychic.rereference_rec` function takes as input the data (using whatever
reference) and a list of electrode indices. The result is a version of the data
referenced to the average signal of the given electrodes:

>>> import psychic
>>> d = psychic.load_bdf(psychic.find_data_path('priming-short.bdf'))
>>> # Mastoid electrodes were 36 and 37:
>>> d_referenced = psychic.rereference_rec(d, [36, 37])
>>> print d_referenced
DataSet with 149504 instances, 40 features [40], 1 classes: [149504], extras: []

Common Average Reference (CAR)
##############################

Another useful reference scheme is the Common Average Reference (CAR). Here, the reference
signal is the average of all EEG electrodes:

>>> # Re-reference to the average of electrodes 0-31:
>>> d_referenced = psychic.rereference_rec(d, range(32))
>>> print d_referenced
DataSet with 149504 instances, 40 features [40], 1 classes: [149504], extras: []

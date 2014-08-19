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

 - ``data`` - the [channels x samples] EEG data
 - ``labels`` - the status channel
 - ``feat_lab`` - the channel names
 - ``ids`` - timestamps for each sample

Referencing
+++++++++++

EEG measures voltage differences between each of the electrodes and some common
reference. The choise of reference can greatly influence your signal-to-noise
ratio (SNR) and the shape the of the EEG in general. It is likely that the EEG
recorder stores the data referenced to some default location. For example,
BioSemi stores it's data referenced to the CMS electrode. When this is the
case, the signal must be re-referenced to use the actual reference electrodes.
When your data contains EOG recordings, it is likely you wish to use a bipolar
referencing scheme for it. 

The :class:`psychic.nodes.EEGMontage` node enables you to specify almost any
referencing scheme imaginable and apply it to your data. Some common examples
are given below.

A commonly used reference is 'linked mastoids', where an electrodes are placed
behind each ear and the average signal of the two electrodes is taken as
reference. 

>>> import psychic
>>> d = psychic.load_bdf(psychic.find_data_path('priming-short.bdf'))
>>> # Mastoid electrodes were EXG1 and EXG2
>>> montage = psychic.nodes.EEGMontage(ref=['EXG1', 'EXG2'])
>>> d_referenced = montage.train_apply(d)
>>> print d_referenced
DataSet with 149504 instances, 41 features [41], 3 classes: [148792, 355, 357], extras: []

Another useful reference scheme is the Common Average Reference (CAR). Here,
the reference signal is the average of all EEG electrodes:

>>> # Speficy all EEG channels (the recording also contains EOG, which
>>> # we don't want to use as reference)
>>> montage = psychic.nodes.EEGMontage(eeg=range(32))
>>> d_referenced = montage.train_apply(d)
>>> print d_referenced
DataSet with 149504 instances, 41 features [41], 3 classes: [148792, 355, 357], extras: []

Linked mastoid reference, horizontal and vertical EOG (bipolar reference), radial
EOG to be calculated and 2 channels that are not connected to anything. After
referencing, drop reference and individual EOG channels.

>>> montage = psychic.nodes.EEGMontage(heog=['EXG3', 'EXG4'], veog=['EXG5', 'EXG6'], calc_reog=True, ref=['EXG1', 'EXG2'], drop=['EXG7', 'EXG8'], drop_ref=True) 
>>> d_referenced = montage.train_apply(d)
>>> print d_referenced.feat_lab
[['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz', 'hEOG', 'vEOG', 'rEOG']]

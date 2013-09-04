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


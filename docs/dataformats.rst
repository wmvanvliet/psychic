Loading from and saving to common dataformats
=============================================

Simple loading of a BDF/EDF file
--------------------------------
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

>>> import psychic
>>> d = psychic.load_bdf(psychic.find_data_path('sine-256Hz.bdf'))
>>> print d
DataSet with 15360 instances, 16 features [16], 2 classes: [8826, 6534], extras: []

In order to have more control over the process, use the
:class:`psychic.BDFReader` class. The BDF fileformat begins with a header in
which various properties of the recording are stored. You can access this
header as a Python dictionary through the ``header`` property of the
:class:`psychic.BDFReader`:

>>> r = psychic.BDFReader(psychic.find_data_path('sine-256Hz.bdf'))
>>> print r.header.keys()
['date_time', 'physical_min', 'record_length', 'local_recording_id', 'physical_max', 'n_records', 'n_samples_per_record', 'label', 'digital_max', 'prefiltering', 'n_channels', 'units', 'local_subject_id', 'transducer_type', 'digital_min']

Once the file is opened, the file can be read all at once using the
``read_all`` method, or in chunks. A BDF file consists of records, which usually
contain one second of data. The ``read`` method returns a `generator <https://wiki.python.org/moin/Generators>`_ that reads one
or more records at the time:

>>> # read 4 records, one by one
>>> chunks = [r.read().next() for i in range(4)]
>>> print len(chunks)
4

>>> # read 10 records at once (a new generator is created here, so the 10 records
>>> # are read from the beginning of the file)
>>> big_chunk = r.read(10).next()

>>> # close the file
>>> r.close()

Note that the most efficient way to concatenate :class:`psychic.DataSet` objects
is to use the :func:`psychic.concatenate` function.

>>> d = psychic.concatenate(chunks)
>>> print d
DataSet with 1024 instances, 16 features [16], 2 classes: [474, 550], extras: []

Writing a BDF file
------------------

Writing a BDF file can be as simple as calling :func:`psychic.save_bdf`:

>>> psychic.save_bdf(d, 'my-file.bdf')

More control over the process can be obtained by using the
:class:`psychic.BDFWriter` class. It takes a dictionary in the same format
as the ``header`` property of :class:`psychic.BDFReader` to use as header values:

>>> w = psychic.BDFWriter('my-file.bdf', header=r.header)
>>> w.close()

Alternatively you can specify a :class:`psychic.DataSet` to extract meta-data from
to construct the header fields:

>>> w = psychic.BDFWriter('my-file.bdf', dataset=d)
>>> w.close()

Or, as a bare minimum, you can specify the number of channels and sample rate:

>>> w = psychic.BDFWriter('my-file.bdf', sample_rate=256, num_channels=10)
>>> w.close()

When a BDF file is opened, :class:`psychic.DataSet` objects can be written to
it using the ``write`` method:

>>> # Create two datasets
>>> d1 = d[:500]
>>> d2 = d[500:]
>>> # Open BDF file
>>> w = psychic.BDFWriter('my-file.bdf', header=r.header)
>>> # Write both datasets
>>> w.write(d1)
>>> w.write(d2)
>>> # Close BDF file
>>> w.close() 

The file on disk is updated after each call to ``write`` so even if the program
crashes halfway, the file will be a valid BDF file.

Processing a BDF file in chunks
-------------------------------

A common usage example of the :class:`psychic.BDFReader` and
:class:`psychic.BDFWriter` is to process a huge BDF file in chunks, writing
each chunk to a new BDF file.

Say the intern has left the sample rate of the EEG recorder to 2048Hz and recorded
two hours of data. The resulting 2 gigabyte BDF file can be downsampled like this::
  r = psychic.BDFReader('huge-file.bdf')
  # update sample rate in the BDF header
  header = r.header.copy()
  header['n_samples_per_record'] = [256 for i in range(len(header['n_samples_per_record']))]
  w = psychic.BDFWriter('resampled-file.bdf', header=header)
  for d in r.read():
      d = psychic.nodes.Resample(256).train_apply(d)
      w.write(d)
  r.close()
  w.close()

Referencing
-----------

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

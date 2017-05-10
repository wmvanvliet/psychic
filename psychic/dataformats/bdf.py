import re, datetime, unittest, logging, struct
import numpy as np
import psychic

bdf_log = logging.getLogger('BDFReader')

class BDFEndOfData(BaseException): pass

class BDFReader:
    '''
    Class to read a large BDF file in chunks.
    '''
    def __init__(self, file):
        '''
        Opens the BDF file. And sets some properties:

         - `self.header`: the header
         - `self.labels`: channel labels
         - `self.sample_rate`: the sample rate

        Parameters
        ----------
        file : string or file
            The filename or filehandle of the BDF file to open.
        '''
        if type(file) == str:
            self.file = open(file, 'rb')
        else:
            self.file = file

        self.header = self.read_header()
        self.labels = self.header['label']
        self.sample_rate = (np.asarray(self.header['n_samples_per_record']) /
            np.asarray(self.header['record_length']))
        self.T = 0.0

    def read_header(self):
        '''
        Read the header of the BDF-file. The header is stored in self.header.
        '''
        f = self.file
        h = self.header = {}
        assert(f.tell() == 0) # check file position
        assert(f.read(8) == b'\xffBIOSEMI')

        # recording info
        h['local_subject_id'] = read_str(f, 80)
        h['local_recording_id'] = read_str(f, 80)

        # parse timestamp
        (day, month, year) = [int(x) for x in re.findall('(\d+)', read_str(f, 8))]
        (hour, minute, sec)= [int(x) for x in re.findall('(\d+)', read_str(f, 8))]
        h['date_time'] = datetime.datetime(year + 2000, month, day, 
            hour, minute, sec)

        # misc
        self.header_nbytes = int(f.read(8))
        format = read_str(f, 44)
        # BioSig toolbox does not set format
        # assert format == '24BIT'
        h['n_records'] = int(f.read(8))
        h['record_length'] = float(f.read(8)) # in seconds
        self.nchannels = h['n_channels'] = int(f.read(4))

        # read channel info
        channels = list(range(h['n_channels']))
        h['label'] = [read_str(f, 16) for n in channels]
        h['transducer_type'] = [read_str(f, 80) for n in channels]
        h['units'] = [read_str(f, 8) for n in channels]
        h['physical_min'] = [int(f.read(8)) for n in channels]
        h['physical_max'] = [int(f.read(8)) for n in channels]
        h['digital_min'] = [int(f.read(8)) for n in channels]
        h['digital_max'] = [int(f.read(8)) for n in channels]
        h['prefiltering'] = [read_str(f, 80) for n in channels]
        h['n_samples_per_record'] = [int(f.read(8)) for n in channels]
        f.read(32 * h['n_channels']) # reserved
        
        assert f.tell() == self.header_nbytes

        self.gain = np.array([(h['physical_max'][n] - h['physical_min'][n]) / 
            float(h['digital_max'][n] - h['digital_min'][n]) for n in channels], 
            np.float)
        self.offset = np.array([h['physical_min'][n] - self.gain[n]*h['digital_min'][n] for n in channels])
        return self.header
    
    def read_record(self, n=1):
        '''
        Read records with data for all channels. 

        Parameters
        ----------
        n : int (default=1)
            The number of records to read at a time.

        Returns
        -------
        records: 2D array (channels x samples)
            The raw 24 bit int values, no gain and offset is applied. If not all
            n records could be read (due to end of file), it returns less. If no
            records could be read, it raises an BDFEndOfData exception.

        See also
        --------
            :func:`psychic.BDFReader.read`
        '''
        h = self.header
        n_channels = h['n_channels']
        n_samp = h['n_samples_per_record']
        assert len(np.unique(n_samp)) == 1, \
            'Sample rates differ for different channels'
        n_samp = n_samp[0]
        result = np.zeros((n_channels, n_samp * n), np.float)

        try:
            for i in range(n):
                for j in range(n_channels):
                    bytes = self.file.read(n_samp * 3)
                    if len(bytes) != n_samp * 3:
                        raise BDFEndOfData
                    result[j, i*n_samp:(i+1)*n_samp] = le_to_int24(bytes)
        except BDFEndOfData:
            if i == 0:
                raise BDFEndOfData
            else:
                return result[j, :i*n_samp]

        return result
    
    def close(self):
        '''
        Closes the BDF file.
        '''
        self.file.close()

    def records(self, n=1):
        '''
        Record generator. 

        Parameters
        ----------
        n : int (default=1)
            The number of records to read at a time.

        Returns
        -------
        records: 2D array (channels x samples)
            The raw 24 bit int values, no gain and offset is applied.

        See also
        --------
            :func:`psychic.BDFReader.all_records`:
            :func:`psychic.BDFReader.read`:
        '''
        try:
            while True:
                yield self.read_record(n)
        except BDFEndOfData:
            pass

    def all_records(self):
        '''
        Reads all records. Returns a 2D array: channels x samples. Note that
        this returns the raw 24 bit int values, no gain and offset is applied.

        See also
        --------
            :func:`psychic.BDFReader.records`:
            :func:`psychic.BDFReader.read_all`:
        '''
        records = list(self.bdf.records())
        rframes = np.hstack(records)
        return rframes


    def read(self, n=1):
        '''
        `:class:psychic.DataSet:` generator

        Parameters
        ----------
        n : int (default=1)
            The number of records to read at a time.

        Returns
        -------
        d : :class:`psychic.DataSet`:
            The records, concatenated into a :class:`psychic.DataSet`:
             - ``d.data`` will be the [channels x samples] EEG data
             - ``d.labels`` will contain the status channel
             - ``d.feat_lab`` will contain the channel names
             - ``d.ids`` will contain timestamps for each sample
             - ``d.data`` will be the [channels x samples] EEG data
             - ``d.labels`` will contain the status channel
             - ``d.feat_lab`` will contain the channel names

            If not all n records could be read (due to end of file), it returns
            less. If no records could be read, it raises an BDFEndOfData
            exception.    

        See also
        --------
            :func:`psychic.BDFReader.all_records`:
        '''
        STATUS = 'Status'

        try:
            while True:
                data = self.read_record(n)
                time = np.arange(data.shape[1]) / self.sample_rate[0] + self.T
                self.T = time[-1] + 1/self.sample_rate[0]

                data_mask = [i for i, lab in enumerate(self.labels) if lab != STATUS]
                status_mask = self.labels.index(STATUS)

                yield psychic.DataSet(
                    data = (data[data_mask,:].T * self.gain[data_mask] + self.offset[data_mask]).T,
                    labels = data[status_mask,:].astype(int) & 0xffff, 
                    ids = time,
                    feat_lab = [self.labels[i] for i in data_mask],
                )
        except BDFEndOfData:
            pass

    def read_all(self):
        '''
        Read all remaining records. Returns a `:class:psychic.DataSet:`

        Returns
        -------
        d : :class:`psychic.DataSet`:
            The records, concatenated into a :class:`psychic.DataSet`:
             - ``d.data`` will be the [channels x samples] EEG data
             - ``d.labels`` will contain the status channel
             - ``d.feat_lab`` will contain the channel names
             - ``d.ids`` will contain timestamps for each sample
             - ``d.data`` will be the [channels x samples] EEG data
             - ``d.labels`` will contain the status channel
             - ``d.feat_lab`` will contain the channel names
             - ``d.ids`` will contain timestamps for each sample
        '''
        records = list(self.read())
        return psychic.concatenate(records, merge_possible_labels=True)

    def __str__(self):
        h = self.header
        return '%s - %s\nChannels [%s] recorded at max %dHz on %s' % \
        (\
        h['local_subject_id'], h['local_recording_id'],
        ', '.join(h['label']), max(h['n_samples_per_record']), h['date_time'],\
        )

def le_to_int24(binary):
    '''Convert groups of 3 bytes (little endian, two's complement) to an
    iterable to a numpy array of 24-bit integers.'''
    if type(binary) == str or type(binary) == bytes:
        binary = np.fromstring(binary, np.uint8)
    else:
        binary = np.asarray(binary, np.uint8)
    int_rows = binary.reshape(-1, 3).astype(np.int32)
    ints = int_rows[:, 0] + (int_rows[:, 1] << 8) + (int_rows[:, 2] << 16)
    ints[ints >= (1 << 23)] -= (1 << 24)
    return ints
    
def int24_to_le(ints):
    '''Convert an interable with 24-bit ints to little endian, two's complement
    numpy array.'''
    uints = np.array(ints, np.int32)
    uints[np.asarray(ints) < 0] -= (1 << 24)
    binary = np.zeros((uints.size, 3), np.uint8)
    binary[:, 0] = (uints & 0xff).flatten()
    binary[:, 1] = ((uints >> 8) & 0xff).flatten()
    binary[:, 2] = ((uints >> 16) & 0xff).flatten()
    return binary.flatten()

def num_to_bytes(num, strlen, encoding='utf-8'):
    return bytes(('%d' % num).ljust(strlen), encoding)

def read_str(file, maxlen, encoding='utf-8'):
    '''Read a string from a binary file.'''
    return file.read(maxlen).decode(encoding).strip()

def str_to_bytes(string, maxlen, encoding='utf-8'):
    '''Format a string into a bytearray'''
    return bytes(string.ljust(maxlen), encoding)

class BDFWriter:
    '''
    Class that writes a stream of :class:`psychic.DataSet` objects to a BDF file.

    Parameters
    ----------
    file : string or file handle
        The file name or file handle of the BDF file to open for writing.

    sample_rate : float
        Sample rate of the data. If not specified, this is inferred form the
        ``header`` and/or ``dataset`` parameter.

    num_channels : int
        Number of channels in the data (not counting the STATUS channel).
        If not specified, this is inferred from the ``header`` and/or
        ``dataset`` parameter.

    header : dict
        A dictionary containing header values as obtained by the
        :class:`psychic.BDFReader` to use as defaults for the header.

    dataset : :class:`psychic.DataSet`
        As an alternative to the ``header`` parameter, vital header fields
        can be determined automatically from a :class:`psychic.DataSet`

    Examples
    --------
    Writing some data to a BDF file::

      d = psychic.DataSet.load('some_data.dat')
      d2 = psychic.DataSet.load('some_more_data.dat')
      bdf = BDFWriter('test.bdf', dataset=d)
      bdf.write_header()
      bdf.write(d)
      bdf.write(d2)
      bdf.close()

    See also
    --------
    :class:`psychic.BDFReader`
    :func:`psychic.load_bdf`
    :func:`psychic.save_bdf`

    '''
    def __init__(self, file, sample_rate=0, num_channels=0, header={}, dataset=None):
        try:
            self.f = open(file, 'wb') if isinstance(file, str) else file
        except:
            raise

        if dataset is not None:
            # Figure out some values from the datafile
            header = header.copy()
            header['n_channels'] = dataset.nfeatures

            if dataset.feat_lab is not None:
                header['label'] = list(dataset.feat_lab[0])

            sample_rate = psychic.get_samplerate(dataset)
            
            record_length = header['record_length'] if 'record_length' in header else 1
            header['n_samples_per_record'] = [int(sample_rate*record_length) for x in range(header['n_channels'])]

        # Use supplied header or defaults
        self.id_code = b'\xffBIOSEMI'
        self.local_subject_id = header['local_subject_id'] if 'local_subject_id' in header else ''
        self.local_recording_id = header['local_recording_id'] if 'local_recording_id' in header else ''

        start_date_time = header['date_time'] if 'date_time' in header else datetime.datetime.now()
        self.start_date = header['start_date'] if 'start_date' in header else start_date_time.strftime('%d.%m.%y')
        self.start_time = header['start_time'] if 'start_time' in header else start_date_time.strftime('%H.%M.%S')

        self.format = header['format'] if 'format' in header else '24BIT'
        self.n_records = header['n_records'] if 'n_records' in header else -1
        self.record_length = header['record_length'] if 'record_length' in header else 1

        n_channels = header['n_channels'] if 'n_channels' in header else num_channels
        assert n_channels > 0, 'Please supply the number of channels.'
        self.n_channels = n_channels

        self.label = header['label'] if 'label' in header else [('channel %d' % (x+1)) for x in range(n_channels)]
        self.transducer_type = header['transducer_type'] if 'transducer_type' in header else ['unknown' for x in range(n_channels)]
        self.units = header['units'] if 'units' in header else ['uV' for x in range(n_channels)]
        self.physical_min = header['physical_min'] if 'physical_min' in header else [-1 for x in range(n_channels)]
        self.physical_max = header['physical_max'] if 'physical_max' in header else [1 for x in range(n_channels)]
        self.digital_min = header['digital_min'] if 'digital_min' in header else [-1000 for x in range(n_channels)]
        self.digital_max = header['digital_max'] if 'digital_max' in header else [1000 for x in range(n_channels)]
        self.prefiltering = header['prefiltering'] if 'prefiltering' in header else ['' for x in range(n_channels)]
        self.n_samples_per_record = header['n_samples_per_record'] if 'n_samples_per_record' in header else [sample_rate for x in range(n_channels)]
        assert len(np.unique(self.n_samples_per_record)) == 1, 'Sample rates differ for different channels'
        assert self.n_samples_per_record[0] > 0, 'Number of samples per record cannot be determined. Please specify a sample rate.'
        self.reserved = header['reserved'] if 'reserved' in header else ['Reserved' for x in range(n_channels)]

        self.records_written = 0
        self.samples_left_in_record = None

        # Append status channel if necessary
        if not 'Status' in self.label:
            self.append_status_channel()

        self.gain = np.array(self.physical_max) - np.array(self.physical_min)
        self.gain = self.gain.astype(np.float)
        self.gain /= np.array(self.digital_max) - np.array(self.digital_min)
        self.inv_gain = 1 / self.gain
        self.offset = np.array(self.physical_min) - self.gain * np.array(self.digital_min)
        self.header_written = False

    def append_status_channel(self):
        self.n_channels = self.n_channels + 1
        self.label.append('Status')
        self.transducer_type.append('')
        self.units.append('')
        self.physical_min.append(0)
        self.physical_max.append(16777215)
        self.digital_min.append(0)
        self.digital_max.append(16777215)
        self.prefiltering.append('')
        self.n_samples_per_record.append(self.n_samples_per_record[0])
        self.reserved.append('')

    def write_header(self):
        """ Write the BDF file header, settings things such as the number of channels and sample rate. """

        # Sanity checks on lengths
        assert len(self.label) == self.n_channels
        assert len(self.transducer_type) == self.n_channels
        assert len(self.units) == self.n_channels
        assert len(self.physical_min) == self.n_channels
        assert len(self.physical_max) == self.n_channels
        assert len(self.digital_min) == self.n_channels
        assert len(self.digital_max) == self.n_channels
        assert len(self.prefiltering) == self.n_channels
        assert len(self.n_samples_per_record) == self.n_channels
        assert len(self.reserved) == self.n_channels

        try:
            # Seek back to the beginning of the file
            self.f.seek(0,0)
            self.f.write(
                struct.pack('8s80s80s8s8s8s44s8s8s4s',
                    self.id_code,
                    str_to_bytes(self.local_subject_id, 80),
                    str_to_bytes(self.local_recording_id, 80),
                    str_to_bytes(self.start_date, 8),
                    str_to_bytes(self.start_time, 8),
                    num_to_bytes(self.n_channels*256 + 256, 8),
                    str_to_bytes(self.format, 44),
                    num_to_bytes(self.n_records, 8), 
                    num_to_bytes(self.record_length, 8),
                    num_to_bytes(self.n_channels, 4),
                )
            )
            
            for label in self.label:
                self.f.write( struct.pack('16s', str_to_bytes(label, 16)) )
            for transducer_type in self.transducer_type:
                self.f.write( struct.pack('80s', str_to_bytes(transducer_type, 80)) )
            for units in self.units:
                self.f.write( struct.pack('8s', str_to_bytes(units, 8)) )
            for physical_min in self.physical_min:
                self.f.write( struct.pack('8s', num_to_bytes(physical_min, 8)) )
            for physical_max in self.physical_max:
                self.f.write( struct.pack('8s', num_to_bytes(physical_max, 8)) )
            for digital_min in self.digital_min:
                self.f.write( struct.pack('8s', num_to_bytes(digital_min, 8)) )
            for digital_max in self.digital_max:
                self.f.write( struct.pack('8s', num_to_bytes(digital_max, 8)) )
            for prefiltering in self.prefiltering:
                self.f.write( struct.pack('80s', str_to_bytes(prefiltering, 80)) )
            for n_samples_per_record in self.n_samples_per_record:
                self.f.write( struct.pack('8s', num_to_bytes(n_samples_per_record, 8)) )
            for reserved in self.reserved:
                self.f.write( struct.pack('32s', str_to_bytes(reserved, 32)) ) 
            self.f.flush()

            # Seek forward to the end of the file
            self.f.seek(0,2)

            self.header_written = True
        except:
            raise

    def write(self, dataset):
        """ Append a Psychic DataSet to the BDF file. Physical values are converted to digital ones using the
        physical/digital_min/max values supplied in the header."""
        self.write_raw(
            psychic.DataSet(
                ((dataset.data.T-self.offset[:-1])*self.inv_gain[:-1]).astype(np.int32).T,
                default=dataset
            )
        )

    def write_raw(self, dataset):
        """ Append a Psychic DataSet to the BDF file. No conversion between physical and digital values are performed. """
        assert len(self.n_samples_per_record) > 0 and self.n_samples_per_record[0] > 0, "Number of samples per record not set!"
        assert dataset.nfeatures == self.n_channels-1, 'Number of channels in dataset does not respond to the number of channels set in the header'
        assert dataset.labels.shape[0] == 1, 'Dimensions of labels must be (samples x 1).idst is used as STATUS channel'

        if not self.header_written:
            self.write_header()

        # Prepend any data left in the buffer from the previous write
        if self.samples_left_in_record is not None:
            dataset = psychic.concatenate([self.samples_left_in_record, dataset], merge_possible_labels=True)
            self.samples_left_in_record = None

        num_channels, num_samples = dataset.data.shape
        for i in range(0, num_samples, self.n_samples_per_record[0]):
            if i+self.n_samples_per_record[0] <= num_samples:
                # Convert data to 24 bit little endian, two's complement
                data = int24_to_le(dataset.data[:, i:i+self.n_samples_per_record[0]])
                self.f.write(data)

                # Bit 20 is used as overflow detection, this code keeps it fixed at '1' (no overflow) 
                status = int24_to_le(dataset.y[i:i+self.n_samples_per_record[0]].astype(np.int) | (1<<19))
                self.f.write(status)

                self.records_written += 1
            else:
                # Store any data left, it will be written the next time write() is called
                self.samples_left_in_record = dataset[i:]

        self.f.flush()

    def close(self):
        """ Write the number of records written in the header and close the BDF file. """
        self.n_records = self.records_written
        self.write_header()
        self.f.close()

def load_bdf(fname):
  '''
  Load a BDF file. BDF files are a variant of EDF files that are produced by
  the BioSemi software. See also the `BDF file specifications
  <http://www.biosemi.com/faq/file_format.htm>`_.

  Parameters
  ----------
  fname : string
    The file name of the BDF file to load.

  Returns
  -------
  d : :class:`psychic.DataSet`
    The loaded data:

      - ``d.data`` will be the [channels x samples] EEG data
      - ``d.labels`` will contain the status channel
      - ``d.feat_lab`` will contain the channel names
      - ``d.ids`` will contain timestamps for each sample
  '''
  with open(fname, 'rb') as f:
      b = BDFReader(f)
      return b.read_all()

def save_bdf(d, fname):
  '''
  Save a :class:`psychic.DataSet` to a BDF file. BDF files are a variant of EDF
  files that are produced by BioSemi software. See also the `BDF file
  specifications <http://www.biosemi.com/faq/file_format.htm>`_.

  Parameters
  ----------
  d : :class:`psychic.DataSet`
    The dataset to save. Must be continous, 2D data.

  fname : string
    The filename of the BDF file to write to. If the file already exists, it is
    overwritten.
  '''
  assert len(d.feat_shape) == 1, 'Expected 2D continuous data'
  nchannels = d.nfeatures
  header={'physical_min': np.floor(np.min(d.data)).repeat(nchannels).tolist(),
          'physical_max': np.ceil(np.max(d.data)).repeat(nchannels).tolist(),
          'digital_min': (-8388608*np.ones(nchannels)).tolist(),
          'digital_max': (8388607*np.ones(nchannels)).tolist()}

  with open(fname, 'wb') as f:
    w = BDFWriter(f, dataset=d, header=header)
    w.write(d)
    w.close()

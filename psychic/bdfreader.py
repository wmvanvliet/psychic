import re, datetime, unittest, logging, warnings, struct
import string
import numpy as np
from golem import DataSet
from psychic import get_samplerate
from markers import biosemi_find_ghost_markers

bdf_log = logging.getLogger('BDFReader')

class BDFEndOfData: pass

class BDFReader:
  '''Simple wrapper to hide the records and read specific number of frames'''
  def __init__(self, file):
    self.bdf = BaseBDFReader(file)
    self.bdf.read_header()
    self.labels = self.bdf.header['label']
    self.sample_rate = (np.asarray(self.bdf.header['n_samples_per_record']) /
      np.asarray(self.bdf.header['record_length']))
    self.buff = self.bdf.read_record()

  def read_nframes_raw(self, nframes):
    while self.buff == None or nframes > self.buff.shape[0]:
      # read more data
      rec = self.bdf.read_record()
      self.buff = np.vstack([self.buff, rec])
    
    # buffer contains enough data
    result = self.buff[:nframes, :]
    self.buff = self.buff[nframes:, :]
    return result

  def read_nframes(self, nframes):
    rframes = self.read_nframes_raw(nframes)
    return rframes * self.bdf.gain + self.bdf.offset

  def read_all(self):
    records = [self.buff] + list(self.bdf.records())
    rframes = np.vstack(records)
    return rframes * self.bdf.gain + self.bdf.offset

class BaseBDFReader:
  def __init__(self, file):
    self.file = file

  def read_header(self):
    '''Read the header of the BDF-file. The header is stored in self.header.'''
    f = self.file
    h = self.header = {}
    assert(f.tell() == 0) # check file position
    assert(f.read(8) == '\xffBIOSEMI')

    # recording info
    h['local_subject_id'] = f.read(80).strip()
    h['local_recording_id'] = f.read(80).strip()

    # parse timestamp
    (day, month, year) = [int(x) for x in re.findall('(\d+)', f.read(8))]
    (hour, minute, sec)= [int(x) for x in re.findall('(\d+)', f.read(8))]
    h['date_time'] = str(datetime.datetime(year + 2000, month, day, 
      hour, minute, sec))

    # misc
    self.header_nbytes = int(f.read(8))
    format = f.read(44).strip()
    # BIOSIG toolbox does not always set format
    #assert format == '24BIT'
    h['n_records'] = int(f.read(8))
    h['record_length'] = float(f.read(8)) # in seconds
    self.nchannels = h['n_channels'] = int(f.read(4))

    # read channel info
    channels = range(h['n_channels'])
    h['label'] = [f.read(16).strip() for n in channels]
    h['transducer_type'] = [f.read(80).strip() for n in channels]
    h['units'] = [f.read(8).strip() for n in channels]
    h['physical_min'] = [float(f.read(8)) for n in channels]
    h['physical_max'] = [float(f.read(8)) for n in channels]
    h['digital_min'] = [int(f.read(8)) for n in channels]
    h['digital_max'] = [int(f.read(8)) for n in channels]
    h['prefiltering'] = [f.read(80).strip() for n in channels]
    h['n_samples_per_record'] = [int(f.read(8)) for n in channels]
    f.read(32 * h['n_channels']) # reserved
    
    assert f.tell() == self.header_nbytes

    self.gain = np.array([(h['physical_max'][n] - h['physical_min'][n]) / 
      float(h['digital_max'][n] - h['digital_min'][n]) for n in channels], 
      np.float32)
    self.offset = np.array(h['physical_min'])

    return self.header
  
  def read_record(self):
    '''
    Read a record with data for all channels, and return an 2D array,
    sampels * channels
    '''
    h = self.header
    n_channels = h['n_channels']
    n_samp = h['n_samples_per_record']
    assert len(np.unique(n_samp)) == 1, \
      'Samplerates differ for different channels'
    n_samp = n_samp[0]
    result = np.zeros((n_samp, n_channels), np.float32)

    for i in range(n_channels):
      bytes = self.file.read(n_samp * 3)
      if len(bytes) <> n_samp * 3:
        raise BDFEndOfData
      result[:, i] = le_to_int24(bytes)

    return result

  def records(self):
    '''
    Record generator.
    '''
    try:
      while True:
        yield self.read_record()
    except BDFEndOfData:
      pass
  
  def __str__(self):
    h = self.header
    return '%s - %s\nChannels [%s] recorded at max %dHz on %s' % \
    (\
    h['local_subject_id'], h['local_recording_id'],
    ', '.join(h['label']), max(h['n_samples_per_record']), h['date_time'],\
    )

def le_to_int24(bytes):
  '''Convert groups of 3 bytes (little endian, two's complement) to an
  iterable to a numpy array of 24-bit integers.'''
  if type(bytes) == str:
    bytes = np.fromstring(bytes, np.uint8)
  else:
    bytes = np.asarray(bytes, np.uint8)
  int_rows = bytes.reshape(-1, 3).astype(np.int32)
  ints = int_rows[:, 0] + (int_rows[:, 1] << 8) + (int_rows[:, 2] << 16)
  ints[ints >= (1 << 23)] -= (1 << 24)
  return ints
  
def int24_to_le(ints):
  '''Convert an interable with 24-bit ints to little endian, two's complement
  numpy array.'''
  uints = np.array(ints, np.int32)
  uints[ints < 0] -= (1 << 24)
  bytes = np.zeros((uints.size, 3), np.uint8)
  bytes[:, 0] = uints & 0xff
  bytes[:, 1] = (uints >> 8) & 0xff
  bytes[:, 2] = (uints >> 16) & 0xff
  return bytes.flatten()

class BDFWriter:
  """ Class that writes a stream of Golem datasets to a BDF file.

  # example usage:
  d = golem.DataSet.load('some_data.dat')
  d2 = golem.DataSet.load('some_more_data.dat')
  bdf = BDFWriter('test.bdf', dataset=d)
  bdf.write_header()
  bdf.write(d)
  bdf.write(d2)
  bdf.close()
  
  """
  
  def __init__(self, file, samplerate=0, num_channels=0, values={}, dataset=None):
    """ Opens a BDF file for writing.

    Required params:
      file - A file (or filename) to open

    Optional params:
      samplerate - Samplerate of the data
      num_channels - Number of channels in the data (not counting the STATUS channel)
      values  - A dictionary containing header values as obtained by the BDFReader to use as defaults
      dataset - As an alternative to the 'values' parameter, vital header fields
            can be determined automatically from a golem DataSet"""

    try:
      self.f = open(file, 'wb') if isinstance(file, str) else file
    except:
      raise

    if dataset != None:
      # Figure out some values from the datafile
      values['n_channels'] = dataset.nfeatures

      if dataset.feat_lab != None:
        values['label'] = dataset.feat_lab

      sample_rate = get_samplerate(dataset)
      
      record_length = values['record_length'] if 'record_length' in values else 1
      values['n_samples_per_record'] = [int(sample_rate*record_length) for x in range(values['n_channels'])]

    # Use supplied values or defaults
    self.id_code = '\xffBIOSEMI'
    self.local_subject_id = values['local_subject_id'] if 'local_subject_id' in values else ''
    self.local_recording_id = values['local_recording_id'] if 'local_recording_id' in values else ''

    start_date_time = datetime.datetime.strptime(values['date_time'], '%Y-%m-%d %H:%M:%S') if 'date_time' in values else datetime.datetime.now()
    self.start_date = start_date_time.strftime('%d.%m.%Y')
    self.start_time = start_date_time.strftime('%H.%M.%S')

    self.format = values['format'] if 'format' in values else '24BIT'
    self.n_records = values['n_records'] if 'n_records' in values else -1
    self.record_length = values['record_length'] if 'record_length' in values else 1

    n_channels = values['n_channels'] if 'n_channels' in values else num_channels
    assert n_channels > 0, 'Please supply the number of channels.'
    self.n_channels = n_channels

    self.label = values['label'] if 'label' in values else [('channel %d' % (x+1)) for x in range(n_channels)]
    self.transducer_type = values['transducer_type'] if 'transducer_type' in values else ['unknown' for x in range(n_channels)]
    self.units = values['units'] if 'units' in values else ['uV' for x in range(n_channels)]
    self.physical_min = values['physical_min'] if 'physical_min' in values else [-1 for x in range(n_channels)]
    self.physical_max = values['physical_max'] if 'physical_max' in values else [1 for x in range(n_channels)]
    self.digital_min = values['digital_min'] if 'digital_min' in values else [-1 for x in range(n_channels)]
    self.digital_max = values['digital_max'] if 'digital_max' in values else [1 for x in range(n_channels)]
    self.prefiltering = values['prefiltering'] if 'prefiltering' in values else ['' for x in range(n_channels)]
    self.n_samples_per_record = values['n_samples_per_record'] if 'n_samples_per_record' in values else [samplerate for x in range(n_channels)]
    assert len(np.unique(self.n_samples_per_record)) == 1, 'Samplerates differ for different channels'
    assert self.n_samples_per_record[0] > 0, 'Number of samples per record cannot be determined. Please specify a samplerate.'
    self.reserved = values['reserved'] if 'reserved' in values else ['' for x in range(n_channels)]

    self.records_written = 0
    self.samples_left_in_record = None

    # Append status channel if necessary
    if not 'Status' in self.label:
      self.append_status_channel()

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
    """ Write the BDF file header, settings things such as the number of channels and samplerate. """
    try:
      # Seek back to the beginning of the file
      self.f.seek(0,0)
      self.f.write(
        struct.pack('8s80s80s8s8s8s44s8s8s4s',
          string.ljust(self.id_code, 8),
          string.ljust(self.local_subject_id, 80),
          string.ljust(self.local_recording_id, 80),
          string.ljust(self.start_date, 8),
          string.ljust(self.start_time, 8),
          '%08d' % (self.n_channels*256 + 256),
          string.ljust(self.format, 44),
          '%08d' % self.n_records,
          '%08d' % self.record_length,
          '%04d' % self.n_channels)
      )
      
      for label in self.label:
        self.f.write( struct.pack('16s', string.ljust(label, 16)) )
      for transducer_type in self.transducer_type:
        self.f.write( struct.pack('80s', string.ljust(transducer_type, 80)) )
      for units in self.units:
        self.f.write( struct.pack('8s', string.ljust(units, 8)) )
      for physical_min in self.physical_min:
        self.f.write( struct.pack('8s', '%08d' % physical_min) )
      for physical_max in self.physical_max:
        self.f.write( struct.pack('8s', '%08d' % physical_max) )
      for digital_min in self.digital_min:
        self.f.write( struct.pack('8s', '%08d' % digital_min) )
      for digital_max in self.digital_max:
        self.f.write( struct.pack('8s', '%08d' % digital_max) )
      for prefiltering in self.prefiltering:
        self.f.write( struct.pack('80s', string.ljust(prefiltering, 80)) )
      for n_samples_per_record in self.n_samples_per_record:
        self.f.write( struct.pack('8s', '%08d' % n_samples_per_record) )
      for reserved in self.reserved:
        self.f.write( struct.pack('32s', string.ljust(reserved, 32)) ) 
      self.f.flush()

      # Seek forward to the end of the file
      self.f.seek(0,2)
    except:
      raise

  def write(self, dataset):
    """ Append a Golem DataSet to the BDF file. Physical values are converted to digital ones using the
    physical/digital_min/max values supplied."""
    num_samples, num_channels = dataset.xs.shape

    inv_gain = np.array( [ (self.digital_max[n] - self.digital_min[n]) / float(self.physical_max[n] - self.physical_min[n]) for n in range(num_channels)] )
    offset = np.array( [self.physical_min[n] for n in range(num_channels)] )
    self.write_raw( DataSet( xs=((dataset.xs-offset)*inv_gain).astype(np.int32), default=dataset ) )

  def write_raw(self, dataset):
    """ Append a Golem DataSet to the BDF file. No conversion between physical and digital values are performed. """
    assert len(self.n_samples_per_record) > 0 and self.n_samples_per_record[0] > 0, "Number of samples per record not set!"
    assert dataset.nfeatures == self.n_channels-1, 'Number of channels in dataset does not respond to the number of channels set in the header'
    assert dataset.ys.shape[1] == 1, 'Dimensions of ys must be (samples x 1). It is used as STATUS channel'

    # Prepend any data left in the buffer from the previous write
    if self.samples_left_in_record != None:
      dataset = self.samples_left_in_record + dataset
      self.samples_left_in_record = None

    num_samples, num_channels = dataset.xs.shape
    for i in range(0, num_samples, self.n_samples_per_record[0]):
      if i+self.n_samples_per_record[0] <= num_samples:
        # Convert data to 24 bit little endian, two's complement
        data = int24_to_le(dataset.xs[i:i+self.n_samples_per_record[0],:].T.flatten())
        self.f.write(data)

        # Bit 20 is used as overflow detection, this code keeps it fixed at '1' (no overflow) 
        status = int24_to_le(dataset.ys[i:i+self.n_samples_per_record[0]].astype(np.int).flatten() | (1<<19))
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


def bdf_dataset(fname):
  warnings.warn('bdf_dataset() is deprecated. Use load_bdf() instead.',
    DeprecationWarning)
  return load_bdf(fname)

def load_bdf(fname):
  STATUS = 'Status'
  f = open(fname, 'rb')
  try:
    b = BDFReader(f)
    frames = b.read_all()
  finally:
    f.close()

  data_mask = [i for i, lab in enumerate(b.labels) if lab != STATUS]
  status_mask = b.labels.index(STATUS)
  feat_lab = [b.labels[i] for i in data_mask]
  sample_rate = b.sample_rate[0]
  ids = (np.arange(frames.shape[0]) / float(sample_rate)).reshape(-1, 1)
  d = DataSet(
    xs=frames[:,data_mask], 
    ys=frames[:,status_mask].reshape(-1, 1).astype(int) & 0xffff, 
    ids=ids, feat_lab=feat_lab, cl_lab=['status'])
  ghosts = biosemi_find_ghost_markers(d.ys.flatten())
  if len(ghosts) > 0:
    logging.getLogger('psychic.bdf_dataset').warning(
      'Found ghost markers: %s' % str(zip(d.ys.flatten()[ghosts], ghosts)))
  return d

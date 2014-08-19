import numpy as np
from ..dataset import DataSet
from ..utils import get_samplerate
from ..trials import slice
from ..filtering import decimate_rec
from basenode import BaseNode
from ..markers import markers_to_events
from collections import deque

class Slice(BaseNode):
  '''
  Extracts fixed-length segments (called trials) of EEG from a recording.
  Segments are sliced based on the onset of some event code.
  
  Given for example an EEG recording which contains two marker codes:
  
  1. Left finger tapping
  2. Right finger tapping
  
  Trials can be extracted in the following manner:
  
  >>> import psychic
  >>> d = psychic.load_bdf(psychic.find_data_path('priming-short.bdf'))
  >>> mdict = {1:'related', 2:'unrelated'}
  >>> slicer = psychic.nodes.Slice(mdict, (-0.2, 1.0))
  >>> trials = slicer.train_apply(d, d)
  >>> print trials
  DataSet with 208 instances, 12280 features [40x307], 2 classes: [104, 104], extras: []
   
  Parameters
  ----------
  markers_to_class : dict
      A dictionary containing as keys the event codes to use as onset of the
      trial and as values a class label for the resulting trials. For example
      ``{1:'left finger tapping', 2:'right finger tapping'}``
  offsets : tuple
      Indicates the time (start, end), relative to the onset of marker, to
      extract as trial. Values are given in seconds.
  '''
  def __init__(self, mark_to_cl, offsets):
    self.mdict, self.offsets = mark_to_cl, np.asarray(offsets)
    BaseNode.__init__(self)

  def train_(self, d):
    self.sample_rate = get_samplerate(d)
    self.offsets_samples = (int(self.offsets[0]*self.sample_rate),
                            int(self.offsets[1]*self.sample_rate))

  def apply_(self, d):
    return slice(d, self.mdict, self.offsets_samples)

class OnlineSlice(Slice):
  '''
  Provides the same functionality as :class:`psychic.nodes.Slice`, but is
  suitable for use in an online scenario.
  '''
  def __init__(self, mark_to_cl, offsets):
    Slice.__init__(self, mark_to_cl, offsets)
    self.reset()

  def reset(self):
    self.buffer = None
    self.event_buffer = []
    self.event_i_buffer = []
    self.cl_lab = sorted(set(self.mdict.values()))

  def apply_(self, d):
    # Initialize datasset
    feat_shape = (d.nfeatures, self.offsets_samples[1]-self.offsets_samples[0])
    slices = DataSet(
      data=np.empty(( feat_shape[0]*feat_shape[1],0 )),
      labels=np.empty(( len(self.cl_lab),0 )),
      ids=np.empty(( 1,0 )),
      feat_shape=feat_shape,
      feat_dim_lab=['channels', 'samples'],
      cl_lab=self.cl_lab
    )

    codes, onsets, durations = markers_to_events(d.labels.flat)
    if self.buffer != None: 
      onsets = list( np.array(onsets) + self.buffer.ninstances )
      d = self.buffer + d
    events = deque(self.event_buffer + zip(codes, onsets))

    while len(events) > 0:
      code, onset = events.popleft()

      if not code in self.mdict:
        continue

      if d.ninstances < onset+self.offsets_samples[1]:
        # Not enough data to extract slice.
        events.appendleft((code, onset))
        break

      # Extract slice
      data = d.data[:, onset+self.offsets_samples[0] : onset+self.offsets_samples[1], np.newaxis]
      labels = np.zeros(( len(self.cl_lab), 1 ))
      labels[self.cl_lab.index(self.mdict[code]), 0] = 1
      ids = np.atleast_2d(d.ids[:,onset+self.offsets_samples[1]-1])
      s = DataSet(data=data, labels=labels, ids=ids, default=slices)
      slices += s

    if len(events) == 0:
      # All slices were extracted
      d = d[max(0, d.ninstances + self.offsets_samples[0]):]
    else:
      first_onset = events[0][1]
      break_point = min(first_onset, first_onset+self.offsets_samples[0])
      events = [(code, onset-break_point) for code, onset in events]
      d = d[break_point:]

    self.event_buffer = list(events)
    self.buffer = d

    return slices

class Decimate(BaseNode):
  '''
  Downsamples the signal by decimation. It takes every nth sample.

  Parameters
  ----------

  factor : int
    Decimation factor. For example, when using a decimation factor of 5, every
    5th sample will be retained and the rest dropped.
  
  max_marker_delay : int (default=0)
    when decimating the signal, markers will sometimes have to move to the next
    available retained sample to avoid being lost. This can result in two
    markers occuring at the same time, in which case one of the markers will be
    be delayed to avoid overlap. this parameter specifies the maximum delay (in
    samples) before an error is generated. 
  '''
  def __init__(self, factor, max_marker_delay=0):
    self.factor = factor
    self.max_marker_delay = max_marker_delay
    BaseNode.__init__(self)

  def apply_(self, d):
    return decimate_rec(d, self.factor, self.max_marker_delay)

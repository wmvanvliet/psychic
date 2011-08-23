import numpy as np
import golem
from ..utils import slice, get_samplerate
from ..filtering import decimate_rec
from golem.nodes import BaseNode
from ..markers import markers_to_events
from collections import deque

class Slice(BaseNode):
  def __init__(self, mark_to_cl, offsets):
    '''
    In contrast to psychic.utils.slice, offsets are specified in *seconds*
    '''
    self.mdict, self.offsets = mark_to_cl, np.asarray(offsets)
    BaseNode.__init__(self)

  def train_(self, d):
    self.sample_rate = get_samplerate(d)
    self.offsets_samples = (int(self.offsets[0]*self.sample_rate),
                            int(self.offsets[1]*self.sample_rate))

  def apply_(self, d):
    return slice(d, self.mdict, (self.offsets * self.sample_rate).astype(int))

class OnlineSlice(Slice):
  def __init__(self, mark_to_cl, offsets):
    Slice.__init__(self, mark_to_cl, offsets)
    self.reset()

  def reset(self):
    self.buffer = None
    self.event_buffer = []
    self.event_i_buffer = []
    self.cl_lab = sorted(set(self.mdict.values()))

  def apply_(self, d):
    slices = d[:0] # initialize to empty dataset
    codes, onsets, durations = markers_to_events(d.Y.flat)
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
      xs = d.xs[onset+self.offsets_samples[0] : onset+self.offsets_samples[1],:]
      Y = np.zeros(( len(self.cl_lab), 1 ))
      Y[self.cl_lab.index(self.mdict[code]), 0] = 1
      I = np.atleast_2d(d.I[:,onset])
      s = golem.DataSet(X=np.atleast_3d(xs).reshape(-1,1),
                        Y=Y, I=I, feat_shape=xs.shape,
                        feat_dim_lab=['samples', 'channels'],
                        cl_lab=self.cl_lab)
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
  def __init__(self, factor, max_marker_delay=0):
    self.factor = factor
    self.max_marker_delay = max_marker_delay
    BaseNode.__init__(self)

  def apply_(self, d):
    return decimate_rec(d, self.factor, self.max_marker_delay)

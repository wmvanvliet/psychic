import numpy as np
from ..dataset import DataSet
from .basenode import BaseNode
from psychic.utils import get_samplerate

class SlidingWindow(BaseNode):
  '''
  Extracts trials from continuous data by applying a sliding window. The
  resulting :class:`psychic.DataSet` *d* will be:
  
   - ``d.data``: [channels x samples x windows]
   - ``d.labels``: Class labels for each window (see ``ref_point`` parameter)
   - ``d.ids``: Timestamps for each window (see ``ref_point`` parameter)

  For example:

  >>> import psychic
  >>> d = psychic.fake.sine(freq=5, nchannels=4, duration=10, sample_rate=100)
  >>> window = psychic.nodes.SlidingWindow(win_size=2, win_step=1)
  >>> d_windowed = window.train_apply(d, d)
  >>> print d_windowed
  DataSet with 9 instances, 800 features [4x200], 1 classes: [9], extras: []

  By default, the center of each window is taken as a reference for timestamps
  and class labels:

  >>> print d_windowed.ids
  [[ 1.  2.  3.  4.  5.  6.  7.  8.  9.]]

  By setting ``ref_point`` to 1.0, the last sample of each window is taken as
  a reference instead:

  >>> window = psychic.nodes.SlidingWindow(win_size=2, win_step=1, ref_point=1.0)
  >>> d_windowed = window.train_apply(d, d)
  >>> print d_windowed.ids
  [[ 1.99  2.99  3.99  4.99  5.99  6.99  7.99  8.99  9.99]]

  Parameters
  ----------
  win_size : float
    The size of the window in seconds.
  win_step : float
    The delay between two consecutive windows in seconds.
  ref_point : float (default=0.5)
    A value between 0 and 1 specifying which sample to take as reference for
    the window. In order to generate a timestamp and class label for the window,
    respectively the time stamp and class label of the reference sample is
    taken.
  '''
  def __init__(self, win_size, win_step, ref_point=.5):
    BaseNode.__init__(self)
    self.win_size = win_size
    self.win_step = win_step
    self.ref_frame = ref_point * self.win_size

  def train_(self, d):
    self.sample_rate = get_samplerate(d)

  def apply_(self, d):
    assert len(d.feat_shape) == 1, 'Expecting 2D continuous data'

    wsize = int(self.win_size * self.sample_rate)
    wstep = int(self.win_step * self.sample_rate)
    refi = int(self.ref_frame * self.sample_rate)
    if refi >= wsize:
        refi = wsize - 1

    data, labels, ids = [], [], []
    for i in range(0, d.ninstances - wsize + 1, wstep):
      win = d[i:i+wsize]
      data.append(win.data[:,:,np.newaxis])
      labels.append(win.labels[:, refi])
      ids.append(win.ids[:, refi])

    if len(data) == 0:
      data = np.zeros((d.nfeatures, wsize, 0)) 
      labels = np.zeros((d.labels.shape[0], 0), dtype=d.labels.dtype) 
      ids = np.zeros((d.ids.shape[0], 0))
    else:
      data = np.concatenate(data, axis=2)
      labels = np.vstack(labels).T
      ids = np.vstack(ids).T

    timestamps = ((np.arange(wsize) - refi) / float(self.sample_rate)).tolist()
    feat_lab = [list(d.feat_lab[0]), timestamps]
      
    return DataSet(data=data, labels=labels, ids=ids, feat_lab=feat_lab,
                   default=d)

class OnlineSlidingWindow(SlidingWindow):
  def __init__(self, win_size, win_step, ref_point=0.5):
    SlidingWindow.__init__(self, win_size, win_step, ref_point)
    self.buffer = None

  def reset(self):
    self.buffer = None

  def apply_(self, d):
    if self.buffer is not None:
      self.buffer = self.buffer + d
    else:
      self.buffer = d

    wstep, wsize = self.win_step, self.win_size
    buff_size = self.buffer.ninstances

    cons = buff_size - buff_size % wstep
    if cons < self.win_size:
      cons = 0
    d, self.buffer = self.buffer[:cons], \
      self.buffer[max(0, cons-wsize + wstep):]

    return SlidingWindow.apply_(self, d)

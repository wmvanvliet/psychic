import numpy as np
from ..dataset import DataSet
from .basenode import BaseNode
from .simple import ApplyOverInstances
from ..utils import spectrogram, sliding_window

class TFC(BaseNode):
  def __init__(self, nfft, win_step):
    def tfc(x):
      return np.array([spectrogram(x[ci,:], nfft, win_step)
        for ci in range(x.shape[0])])

    BaseNode.__init__(self)
    self.nfft, self.win_step = nfft, win_step
    self.n = ApplyOverInstances(tfc)
  
  def apply_(self, d):
    assert len(d.feat_shape) == 2 # [channels x samples]
    if d.feat_dim_lab is not None:
      assert d.feat_dim_lab == ['channels', 'time']

    tfc = self.n.apply(d)
    feat_dim_lab = ['channels', 'time', 'frequency']

    if d.feat_lab is not None:
      old_time = d.feat_lab[1]
      time = np.mean(sliding_window(old_time, self.nfft, self.win_step), axis=1)
      dt = np.mean(np.diff(old_time))
      dt = (np.max(old_time) - np.min(old_time)) / len(old_time)
      freqs = np.fft.fftfreq(self.nfft, dt) 
      freqs = [abs(i) for i in freqs[:self.nfft/2 + 1]]
      channels = d.feat_lab[0]
      feat_lab = [channels, time.tolist(), freqs]
    else:
      feat_lab = None

    return DataSet(feat_dim_lab=feat_dim_lab, feat_lab=feat_lab, 
      default=tfc)

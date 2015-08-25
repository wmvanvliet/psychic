import numpy as np
from golem import DataSet
from golem.nodes import BaseNode
from ..utils import sliding_window_indices

class SlidingWindow(BaseNode):
  def __init__(self, win_size, win_step, ref_point=.5):
    BaseNode.__init__(self)
    self.win_size = win_size
    self.win_step = win_step
    self.ref_frame = int(float(ref_point) * (self.win_size - 1))

  def reset(self):
    self.buffer = None

  def apply_(self, d):
    wsize, wstep, refi = self.win_size, self.win_step, self.ref_frame

    xs, ys, ids = [], [], []
    for i in range(0, d.ninstances - wsize + 1, wstep):
      win = d[i:i+wsize]
      xs.append(win.nd_xs)
      ys.append(win.ys[refi])
      ids.append(win.ids[refi])

    if len(xs) == 0:
      xs = np.zeros((0, wsize * d.nfeatures)) 
      feat_shape = (wsize, d.nfeatures)
      ys = np.zeros((0, d.nclasses)) 
      ids = np.zeros((0, d.ids.shape[1]))
    else:
      xs = np.asarray(xs)
      feat_shape = xs.shape[1:]
      xs = xs.reshape(xs.shape[0], -1)
      ys = np.asarray(ys)
      ids = np.asarray(ids)

    return DataSet(xs=xs, feat_shape=feat_shape, ys=ys, ids=ids, default=d)

class OnlineSlidingWindow(SlidingWindow):
  def __init__(self, win_size, win_step, ref_point=0.5):
    SlidingWindow.__init__(self, win_size, win_step, ref_point)
    self.buffer = None

  def apply_(self, d):
    if self.buffer != None:
      self.buffer = self.buffer + d
    else:
      self.buffer = d

    wstep, wsize = self.win_step, self.win_size
    buff_size = self.buffer.ninstances

    cons = buff_size - buff_size % self.win_step
    if cons < self.win_size:
      cons = 0
    d, self.buffer = self.buffer[:cons], \
      self.buffer[max(0, cons-wsize + wstep):]
    return SlidingWindow.apply_(self, d)

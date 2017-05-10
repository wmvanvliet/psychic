import numpy as np
from .basenode import BaseNode
from ..dataset import DataSet

class ApplyOverInstances(BaseNode):
  def __init__(self, mapping):
    BaseNode.__init__(self)
    self.mapping = mapping

  def apply_(self, d):
    instances = np.asarray(list(map(self.mapping, np.rollaxis(d.data, -1))))
    data = np.rollaxis(instances, 0, len(instances.shape))
    return DataSet(data, default=d)

  def __str__(self):
    return '%s (with mapping "%s")' % (self.name, self.mapping.__name__)

class ApplyOverFeats(BaseNode):
  def __init__(self, mapping):
    BaseNode.__init__(self)
    self.mapping = mapping

  def apply_(self, d):
    data = np.apply_along_axis(self.mapping, 0, d.data.reshape(-1, d.ninstances))
    data = data.reshape(d.data.shape[:-1] + (-1,))
    return DataSet(data, default=d)

  def __str__(self):
    return '%s (with mapping "%s")' % (self.name, self.mapping.__name__)

class ZScore(BaseNode):
  def train_(self, d):
    self.mean = np.atleast_2d(np.mean(d.data, axis=1)).T
    self.std = np.atleast_2d(np.std(d.data, axis=1)).T

  def apply_(self, d):
    return DataSet((d.data - self.mean) / self.std, default=d)

import numpy as np
from basenode import BaseNode
from ..dataset import DataSet

class ApplyOverInstances(BaseNode):
  def __init__(self, mapping):
    BaseNode.__init__(self)
    self.mapping = mapping

  def apply_(self, d):
    instances = np.asarray(map(self.mapping, np.rollaxis(d.data, -1)))
    data = np.rollaxis(instances, 0, len(instances.shape))
    return DataSet(data, default=d)

  def __str__(self):
    return '%s (with mapping "%s")' % (self.name, self.mapping.__name__)

class FeatMap(ApplyOverInstances): pass

class ApplyOverFeats(BaseNode):
  def __init__(self, mapping):
    BaseNode.__init__(self)
    self.mapping = mapping

  def apply_(self, d):
    data = np.apply_along_axis(self.mapping, d.data, axis=-1)
    return DataSet(data, default=d)

  def __str__(self):
    return '%s (with mapping "%s")' % (self.name, self.mapping.__name__)

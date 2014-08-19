import unittest, operator
import numpy as np
from ..utils import sliding_window
from ..nodes import SlidingWindow, OnlineSlidingWindow
from ..dataset import DataSet
from ..helpers import to_one_of_n
from scipy import signal

class TestWindowNode(unittest.TestCase):
  def setUp(self):
    data = np.arange(300).reshape(-1, 3).T
    labels = np.linspace(0, 3, 100, endpoint=False).astype(int)
    self.d = DataSet(data=data, labels=to_one_of_n(labels))

  def test_sw(self):
    d = self.d
    for wsize in [4, 10]:
      for wstep in [2, 5]:
        for ref_point in [0, .5, 1]:
          sw = SlidingWindow(wsize, wstep, ref_point)
          d2 = sw.train_apply(d)

          # test shapes
          self.assertEqual(d2.feat_shape, (d.nfeatures, wsize))

          # test xs
          offset = wstep * d.nfeatures
          max_offset = (d.ninstances - wsize) * d.nfeatures
          base = np.arange(0, max_offset+1, offset)
          detail = np.arange(d.nfeatures * wsize).reshape(wsize, d.nfeatures).T
          target = detail[:,:,np.newaxis] + base
          np.testing.assert_equal(target, d2.data)

          # test ids
          np.testing.assert_equal(np.diff(d2.ids[0,:]), wstep)

          refi = int(wsize * ref_point)
          if refi >= wsize:
              refi = wsize - 1
          self.assertEqual(d2.ids[0, 0], refi)

          # test labels
          np.testing.assert_equal(d.labels[:, d2.ids.flatten()], d2.labels)

  def test_osw(self):
    d = self.d
    for wsize in [4, 10]:
      for wstep in [2, 5]:
        sw = SlidingWindow(10, 5)
        osw = OnlineSlidingWindow(10, 5)

        wins = []
        stream = d
        while len(stream) > 0:
          head, stream = stream[:4], stream[4:]
          wins.append(osw.train_apply(head))

        self.assertEqual(sw.train_apply(d), reduce(operator.add, wins))

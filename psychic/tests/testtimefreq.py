import unittest
from ..utils import spectrogram
from ..trials import slice
from ..nodes import TFC
from ..dataset import DataSet

import numpy as np

FS = 256.

class TestTFC(unittest.TestCase):
  def setUp(self):
    data = np.array([np.sin(i * 4 * 60 * np.linspace(0, np.pi * 2, 60 * FS)) 
      for i in range(16)])
    labels = np.zeros(data.shape[1], dtype=np.int)
    labels[[1000, 2000, 3000, 4000]] = 1
    ids = np.arange(data.shape[1]) / FS

    self.d = slice(DataSet(data, labels, ids), {1:'fake'}, [-512, 512])
    
  def test_setup(self):
    d = self.d
    self.assertEqual(d.feat_shape, (16, 1024))
    self.assertEqual(d.nclasses, 1)
    self.assertEqual(d.ninstances, 4)

  def test_tfc(self):
    d = self.d
    w_size, w_step = 64, 32
    tfc = TFC(w_size, w_step)
    tfc.train(d)
    td = tfc.apply(d)

    nwindows = int(np.floor((d.feat_shape[1] - w_size + w_step) / 
      float(w_step)))
    self.assertEqual(td.feat_shape, (d.feat_shape[0], nwindows, w_size/2+1))
    self.assertEqual(td.nclasses, d.nclasses)
    self.assertEqual(td.ninstances, d.ninstances)

    for ci in range(td.feat_shape[0]):
      a = td.data[ci,:,:,0]
      b = spectrogram(d.data[ci,:,0], w_size, w_step)
      np.testing.assert_equal(a, b)
    self.assertEqual(td.feat_dim_lab, ['channels', 'time', 'frequency'])

    time = td.feat_lab[1]
    np.testing.assert_almost_equal(time, 
      np.linspace((-512 + w_step)/FS, (512 - w_size)/FS, len(time)), 1)

    freq = td.feat_lab[2]
    np.testing.assert_almost_equal(freq, np.arange(32 + 1) * 4, 1)

    self.assertEqual(td.feat_lab[0], d.feat_lab[0])

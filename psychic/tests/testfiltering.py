import unittest, os, operator
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from ..dataset import DataSet
from ..plots import plot_timeseries
from .. import filtering
from ..nodes import OnlineFilter, Winsorize
from functools import reduce

class TestResample(unittest.TestCase):
  def setUp(self):
    data = np.arange(1000).reshape(2, -1)
    labels = np.zeros((1, 500), dtype=np.int)
    labels[0, ::4] = 2
    self.d = DataSet(data=data, labels=labels)

  def test_resample(self): 
    d = self.d
    d2 = filtering.resample_rec(d, .5)
    self.assertEqual(d2.ninstances, d.ninstances / 2)
    self.assertEqual(d2.nfeatures, d.nfeatures)
    self.assertEqual(d2.feat_lab, d.feat_lab)
    self.assertEqual(d2.cl_lab, d.cl_lab)
    self.assertEqual(d2.feat_shape, d.feat_shape)
    np.testing.assert_equal(d2.labels[0, ::2], 2)
    self.assertEqual(np.mean(np.diff(d2.ids, axis=1)), 2)

    # Testing the decimation itself is more difficult due to boundary
    # artifacts, and is the responsibility of Scipy.
    # We do a rough test that it should be similar to naive resampling:
    self.assertTrue(np.std((d2.data - d.data[:,::2])[:, 100:-100]) < 0.6)

  def test_overlapping_markers(self):
    d = self.d
    # test overlapping markers
    self.assertRaises(AssertionError, filtering.decimate_rec, d, 5)


class TestDecimate(unittest.TestCase):
  def setUp(self):
    data = np.arange(100).reshape(2, -1)
    labels = np.zeros((1, 50), dtype=np.int)
    labels[0,::4] = 2
    self.d = DataSet(data, labels=labels)

  def test_aa(self):
    # Create signal with a LF and a HF part. HF should cause aliasing
    data = np.zeros(128)
    data[[-2, -3]] = 4 # HF
    data[8] = 1 # LF
    data = np.fft.irfft(data) + 1

    labels = np.zeros(data.shape, dtype=np.int)
    labels[::4] = 2
    d = DataSet(data=data, labels=labels)

    d2 = filtering.decimate_rec(d, 2)
    self.assertEqual(d2.ninstances, d.ninstances / 2)
    self.assertEqual(d2.nfeatures, d.nfeatures)
    self.assertEqual(d2.feat_lab, d.feat_lab)
    self.assertEqual(d2.cl_lab, d.cl_lab)
    self.assertEqual(d2.feat_shape, d.feat_shape)
    np.testing.assert_equal(d2.labels[0, ::2], 2)
    self.assertEqual(np.mean(np.diff(d2.ids, axis=1)), 2)

    self.assertEqual(np.argsort(np.abs(np.fft.rfft(d.data[0, ::2])))[-2],
      2, 'Without the AA-filter the f=1./2 has most power')
    self.assertEqual(np.argsort(np.abs(np.fft.rfft(d2.data[0, :])))[-2],
      8, 'With the AA-filter, f=1./8 has most power.')

  def test_decimate(self):
    d = self.d
    d2 = filtering.decimate_rec(d, 2)
    self.assertEqual(d2.ninstances, d.ninstances / 2)
    self.assertEqual(d2.nfeatures, d.nfeatures)
    self.assertEqual(d2.feat_lab, d.feat_lab)
    self.assertEqual(d2.cl_lab, d.cl_lab)
    self.assertEqual(d2.feat_shape, d.feat_shape)
    np.testing.assert_equal(d2.labels[:,::2], np.ones((1, 13)) * 2)
    self.assertEqual(np.mean(np.diff(d2.ids, axis=1)), 2)

  def test_overlapping_markers(self):
    d = self.d
    # test overlapping markers
    self.assertRaises(AssertionError, filtering.decimate_rec, d, 5)

class TestFilter(unittest.TestCase):
  def setUp(self):
    np.random.seed(0)
    self.d = DataSet(np.random.rand(4, 400))

  def test_nop(self):
    b, a = np.array([0, 1]), np.array([1])
    self.assertEqual(filtering.filtfilt_rec(self.d, (b, a)), self.d)

  def test_lp(self):
    b, a = signal.iirfilter(4, [.1], btype='low')
    df = filtering.filtfilt_rec(self.d, (b, a))
    spec = np.abs(np.fft.rfft(df.data, axis=1))

    # verify that there is more power in the lowest 10%
    pass_p = np.mean(spec[:, :self.d.ninstances // 10], axis=1)
    stop_p = np.mean(spec[:, self.d.ninstances // 10:], axis=1)
    self.assertTrue(((pass_p/stop_p) > 20).all())

  def test_hp(self):
    b, a = signal.iirfilter(6, [.9], btype='high')
    df = filtering.filtfilt_rec(self.d, (b, a))
    # only test for zero mean
    np.testing.assert_almost_equal(np.mean(df.data, axis=1), np.zeros(4), 3)

class TestOnlineFilter(unittest.TestCase):
  def test_online_filter(self):
    N = 200
    WIN = 50

    d = DataSet(np.random.rand(3, N) + 100)
    d0, stream = d[:10], d[10:]

    def filt_design_f(sr):
      return signal.iirfilter(4, [.01, .2])

    of = OnlineFilter(filt_design_f)
    of.train(d0) # get sampling rate for filter design

    store = []
    tail = stream
    while len(tail) > 0:
     head, tail = tail[:WIN], tail[WIN:]
     store.append(of.apply(head))
    filt_d = reduce(operator.add, store)

    b, a = of.filter
    np.testing.assert_equal(filt_d.data, signal.lfilter(b, a, stream.data, axis=1))

class TestWinsorizing(unittest.TestCase):
  def setUp(self):
    data = np.random.rand(100, 5) + np.arange(5)
    data[10, :] = 10
    data[11, :] = -10
    self.d = DataSet(data.T)

  def test_nop(self):
    d = self.d
    nop = Winsorize([0, 1]).train_apply(d)
    self.assertEqual(nop, d)

  def test_minimal(self):
    d = self.d
    wd = Winsorize([.01, .99]).train_apply(d)
    self.assertTrue(np.all((wd.data == d.data)[:, :10]))
    self.assertTrue(np.all((wd.data != d.data)[:, 10:12]))
    self.assertTrue(np.all((wd.data == d.data)[:, 12:]))

def ewma_ref(x, alpha, v0=0):
  x = np.atleast_1d(x).flatten()
  result = np.zeros(x.size + 1)
  result[0] = v0

  for i in range(1, x.size + 1):
    result[i] = (1.- alpha) * result[i-1] + alpha * x[i-1]
  return result[1:]

class TestMAs(unittest.TestCase):
  def setUp(self):
    np.random.seed(3)
    self.s = np.cumsum(np.random.randn(1000)) + np.random.rand(1000) + 40
    self.s[400] = 100

  def test_ma(self):
    for n in [2, 10, 60]:
      np.testing.assert_almost_equal(
        signal.lfilter(np.ones(n), float(n), self.s), filtering.ma(self.s, n))

  def test_emwa(self):
    s = self.s
    for alpha in [.001, .01, .05, .1]:
      np.testing.assert_almost_equal(
        filtering.ewma(s, alpha), ewma_ref(s, alpha))

    for alpha in [.001, .01, .05, .1]:
      for v0 in [0, .01, 4]:
        np.testing.assert_almost_equal(
          filtering.ewma(s, alpha, v0), ewma_ref(s, alpha, v0))

import unittest
import numpy as np
from numpy import linalg as la
from ..dataset import DataSet
from ..trials import concatenate_trials

from ..nodes.spatialfilter import *

class TestBaseSpatialFilter(unittest.TestCase):
  def setUp(self):
    # build dataset with artificial trials
    dtrial = DataSet( np.random.randn(32, 128, 10) )
      
    # derive cov-based dataset
    covs = np.concatenate([np.cov(dtrial.data[:,:,t])[:,:,np.newaxis]
                           for t in range(dtrial.ninstances)], axis=2)
    dcov = DataSet(data=covs, default=dtrial)

    # construct plain dataset (without trials) based on dtrial
    dplain = concatenate_trials(dtrial)

    self.dplain = dplain
    self.dtrial = dtrial
    self.dcov = dcov

  def test_plain(self):
    d = self.dplain
    f = BaseSpatialFilter(ftype=PLAIN)
    f.W = np.random.randn(32, 4)

    self.assertEqual(f.get_nchannels(d), 32)
    np.testing.assert_equal(f.get_cov(d), cov0(d.data))
    np.testing.assert_equal(f.sfilter(d).data, np.dot(f.W.T, d.data))

  def test_trial(self):
    dtrial = self.dtrial
    f = BaseSpatialFilter(ftype=TRIAL)
    f.W = np.random.randn(32, 4)

    # test extraction of number of channels
    self.assertEqual(f.get_nchannels(dtrial), 32)

    # Test that the covariances are correctly extracted. The devision by n-1
    # causes some small differences.
    np.testing.assert_almost_equal(f.get_cov(dtrial), 
      cov0(concatenate_trials(dtrial).data), decimal=2)

    # verify that the mapping is applied correctly
    np.testing.assert_equal(f.sfilter(dtrial).data, 
      np.concatenate([np.dot(f.W.T, dtrial.data[:,:,t])[:,:,np.newaxis]
                      for t in range(dtrial.ninstances)], axis=2)
    )

  def test_cov(self):
    dtrial = self.dtrial
    dcov = self.dcov

    f = BaseSpatialFilter(ftype=COV)
    f.W = np.random.randn(32, 4)

    # test extraction of number of channels
    self.assertEqual(f.get_nchannels(dcov), 32)

    # test that the covariances are correctly extracted
    np.testing.assert_equal(f.get_cov(dcov), np.mean(dcov.data, axis=2))

    # verify that the mapping is applied correctly
    target = np.concatenate(
        [np.cov(np.dot(f.W.T, dtrial.data[:,:,t]))[:,:,np.newaxis]
         for t in range(dtrial.ninstances)], axis=2)

    np.testing.assert_almost_equal(f.sfilter(dcov).data, target)
      

class TestSpatialFilters(unittest.TestCase):
  def setUp(self):
    np.random.seed(0)

  def test_cov0(self):
    data = np.dot(np.random.rand(4,4), np.random.rand(4, 1000))
    data = (data.T - np.mean(data, axis=1)).T
    np.testing.assert_almost_equal(cov0(data), np.cov(data, bias=1))

  def test_select_channels(self):
    data = np.random.rand(10, 40)
    for keep in [[0, 1, -3, 2], (np.arange(10) % 2 == 0).astype(bool)]:
      np.testing.assert_equal(
        np.dot(select_channels(data.shape[0], keep).T, data), 
        data[keep,:]
    )

  def test_deflate(self):
    np.random.seed(0)

    # make centered xs, with 2 big amplitude channels at the end.
    data = np.dot(np.eye(4), np.random.randn(4, 1000))
    data = np.vstack([data, np.random.randn(2, data.shape[1]) * 20])

    # spread some influence of the big amplitude channels.
    A = np.eye(6)
    A[-2:,:-2] = np.random.rand(2, 4)
    data_mix = np.dot(A.T, data)
    data_mix = (data_mix.T - np.mean(data_mix, axis=1)).T

    # Verify that it undoes the mixing. I suspect that the poor numerical 
    # precision is the result of random correlations in xs.
    sig = cov0(data_mix)
    sig_S = cov0(data)

    W = deflate(sig, [False, False, False, False, True, True])

    np.testing.assert_almost_equal(reduce(np.dot, [W.T, sig, W]), 
      cov0(data[:4, :]), decimal=2)
    np.testing.assert_allclose(np.dot(W.T, data_mix), data[:-2,:], atol=0.23)

  def test_car(self):
    data = np.random.rand(4, 10)
    W = car(data.shape[0])
    self.assert_(np.allclose(np.dot(W.T, data), data - np.mean(data, axis=0)))

  def test_outer_n(self):
    np.testing.assert_equal(outer_n(1), [0])
    np.testing.assert_equal(outer_n(2), [0, -1])
    np.testing.assert_equal(outer_n(3), [0, 1, -1])
    np.testing.assert_equal(outer_n(6), [0, 1, 2, -3, -2, -1])

  def test_whitening(self):
    data = np.random.randn(5, 100)
    W = whitening(cov0(data))
    data2 = np.dot(W.T, data)
    self.assertEqual(data2.shape, data.shape)
    np.testing.assert_almost_equal(cov0(data2), np.eye(5))

  def test_whitening_lowrank(self):
    data = np.dot(np.random.rand(5, 3), np.random.randn(3, 100))
    W = whitening(cov0(data))
    data2 = np.dot(W.T, data)
    np.testing.assert_almost_equal(cov0(data2), np.eye(3))

  def test_sym_whitening(self):
    data = np.random.randn(5, 100)
    W = sym_whitening(np.cov(data))
    data2 = np.dot(W.T, data)

    # test whitening property
    self.assertEqual(data2.shape, data.shape)
    np.testing.assert_almost_equal(np.cov(data2), np.eye(5))

    # test symmetry
    np.testing.assert_almost_equal(W, W.T)

  def test_csp(self):
    xa = (np.random.randn(100, 4) * np.array([1, 1, 1, 3])).T
    xb = (np.random.randn(100, 4) * np.array([1, 1, 1, .1])).T

    # create low-rank data
    A = np.random.rand(4, 8)
    xa = np.dot(A.T, xa)
    xb = np.dot(A.T, xb)

    sig_a = cov0(xa)
    sig_b = cov0(xb)

    for m in range(2, 6):
      W = csp(sig_a, sig_b, m)
      self.assertEqual(W.shape, (8, min(m, 4)))
      D1 = cov0(np.dot(W.T, xa))
      D2 = cov0(np.dot(W.T, xb))

      np.testing.assert_almost_equal(D1 + D2, np.eye(W.shape[1]), 
        err_msg='Joint covariance is not the identity matrix.')
      np.testing.assert_almost_equal(np.diag(np.diag(D1)), D1,
        err_msg='Class covariance is not diagonal.')
      np.testing.assert_almost_equal(np.diag(D1), np.sort(np.diag(D1)),
        err_msg='Class variance is not ascending.')

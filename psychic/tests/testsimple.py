import unittest
import numpy as np
from .. import DataSet, fake
from ..nodes import ApplyOverInstances, ApplyOverFeats, ZScore

class TestApplyOverInstances(unittest.TestCase):
    def setUp(self):
        self.d = DataSet(data=np.arange(100).reshape(5, 2, 10),
                         labels=np.ones(10))

    def test_map(self):
        d = self.d
        d2 = ApplyOverInstances(lambda x: x * 2).apply(d)
        self.assertEqual(d, DataSet(data=d2.data/2, default=d))

    def test_less_features(self):
        d = self.d
        d2 = ApplyOverInstances(lambda x: np.mean(x.flat)).apply(d)
        np.testing.assert_equal(
                d2.data,
                np.atleast_2d(np.mean(np.mean(d.data, axis=0), axis=0))
        )

    def test_nd_feat(self):
        d = self.d
        d2 = ApplyOverInstances(lambda x: x[:3, :]).apply(d)
        np.testing.assert_equal(d2.data, d.data[:3])

    def test_str(self):
        self.assertEqual(str(ApplyOverInstances(lambda x: x)),
                         'ApplyOverInstances (with mapping "<lambda>")')

class TestApplyOverFeats(unittest.TestCase):
    def setUp(self):
        self.d = DataSet(data=np.arange(100).reshape(5, 2, 10),
                         labels=np.ones(10))

    def test_map(self):
        d = self.d
        d2 = ApplyOverFeats(lambda x: np.sort(x)).apply(d)
        self.assertEqual(d2, DataSet(data=np.sort(d.data, axis=1), default=d))

    def test_str(self):
        self.assertEqual(str(ApplyOverFeats(lambda x: x)),
                         'ApplyOverFeats (with mapping "<lambda>")')

class TestZScore(unittest.TestCase):
    def setUp(self):
        self.d = fake.gaussian_dataset([40, 40, 40])
        
    def test_zscore(self):
        '''Test ZScore properties'''
        z = ZScore()
        zd = z.train_apply(self.d, self.d)

        # test for mean==0 and std==1
        np.testing.assert_almost_equal(np.mean(zd.data, axis=1), 0)
        np.testing.assert_almost_equal(np.std(zd.data, axis=1), 1)

        # test inverse
        zd_inv_data = zd.data * z.std + z.mean
        np.testing.assert_almost_equal(zd_inv_data, self.d.data)

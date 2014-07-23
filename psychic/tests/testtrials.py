import unittest, logging
import numpy as np
from ..dataset import DataSet
from .. import trials

class TestSlice(unittest.TestCase):
    def setUp(self):
        data = np.arange(40).reshape(2, -1)
        labels = np.zeros((1, 20))
        labels[0, [0, 2, 16]] = 1
        labels[0, [4, 12, 19]] = 2
        ids = np.r_[np.atleast_2d(np.arange(20)), np.ones((1,20), int)]

        self.d = DataSet(data, labels, ids)

    def test_windows(self):
        data = np.random.rand(5, 20)
        labels = np.zeros((1, 20))
        labels[0, [2, 16]] = 1
        labels[0, [4, 12]] = 2

        d2 = trials.slice(DataSet(data, labels), {1:'b', 2:'a'}, offsets=[-2, 4])

        np.testing.assert_equal(d2.ids.flatten(), [2, 4, 12, 16])

        #win_base = np.arange(12).reshape(-1, 2)
        for i, off in enumerate([2, 4, 12, 16]):
            np.testing.assert_equal(d2.data[:,:,i], data[:,off-2:off+4])

    def test_labels(self):
        data = np.random.rand(5, 20)
        labels = np.zeros((1, 20))
        labels[0, [2, 16]] = 1
        labels[0, [4, 12]] = 2
        labels[0, [8]] = 3
        labels[0, [9]] = 4
        d2 = trials.slice(DataSet(data, labels), {1:'b', 2:'a', 3:'c', 4:'c'}, 
            offsets=[0, 2])
        self.assertEqual(d2.cl_lab, ['a', 'b', 'c'])
        np.testing.assert_equal(d2.get_class(0).ids.flatten(), [4, 12])
        np.testing.assert_equal(d2.get_class(1).ids.flatten(), [2, 16])
        np.testing.assert_equal(d2.get_class(2).ids.flatten(), [8, 9])

    def test_few_trials(self):
        data = np.random.rand(5, 20)
        labels = np.zeros((1, 20))

        mdict = {1:'b', 2:'a'}
    
        ds = trials.slice(DataSet(data, labels.copy()), mdict, [0, 2])
        self.assertEqual(ds.ninstances_per_class, [0, 0])
        
        labels[0, 5] = 1
        ds = trials.slice(DataSet(data, labels.copy()), mdict, [0, 2])
        self.assertEqual(ds.ninstances_per_class, [0, 1])

        labels[0, 5] = 2
        ds = trials.slice(DataSet(data, labels.copy()), mdict, [0, 2])
        self.assertEqual(ds.ninstances_per_class, [1, 0])

        labels[0, 6] = 1
        ds = trials.slice(DataSet(data, labels.copy()), mdict, [0, 2])
        self.assertEqual(ds.ninstances_per_class, [1, 1])

    def test_bounds(self):
        data = np.random.rand(5, 20)
        labels = np.zeros((1, 20))
        labels[0, [3, 20-3]] = 1
        labels[0, [4, 20-4]] = 2
        logging.getLogger('psychic.trials.slice').setLevel(logging.ERROR)
        ds = trials.slice(DataSet(data, labels), {1:'bad', 2:'good'}, [-4, 4])
        self.assertEqual(ds.ninstances, 2)
        np.testing.assert_equal(ds.ids.flatten(), [4, 20-4])
        logging.getLogger('psychic.trials.slice').setLevel(logging.WARNING)

    def test_feat_labs(self):
        data = np.random.rand(2, 20)
        labels = np.zeros((1, 20))
        labels[0, 10] = 1
        ids = np.arange(20) / 10.
        d = DataSet(data, labels, ids, feat_lab=[['a', 'b']])
        ds = trials.slice(d, {1:'hit'}, [-2, 2])

        np.testing.assert_equal(ds.feat_lab[1], np.arange(-2, 2)/10.)
        self.assertEqual(ds.feat_lab[0], d.feat_lab[0])

    def test_nd_ids(self):
        data = np.random.rand(2, 20)
        labels = np.zeros((1, 20))
        labels[0, [10, 12]] = 1
        ids = np.r_[np.atleast_2d(np.arange(20)), np.ones((1, 20))]
        ds = trials.slice(DataSet(data, labels, ids=ids), {1:'hit'}, [-2, 2])
        np.testing.assert_equal(ds.ids, [[10, 12], [1, 1]])

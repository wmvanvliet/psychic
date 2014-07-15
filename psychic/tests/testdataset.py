from ..dataset import DataSet
from ..helpers import to_one_of_n
import numpy as np
import unittest

class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(5,3,10)
        self.labels = np.atleast_2d(np.arange(5, dtype=np.int).repeat(2))
        self.cl_lab = ['cl_a', 'cl_b', 'cl_c', 'cl_d', 'cl_e']
        self.feat_lab = [['a', 'b', 'c', 'd', 'e'], [1.0,2.0,3.0]]
        self.feat_dim_lab = ['channels', 'time']
        self.ids = np.atleast_2d(np.arange(10)/10.)
        self.extra = {'foo': 'bar'}
        self.d = DataSet(
            data=self.data,
            labels=self.labels,
            cl_lab=self.cl_lab,
            feat_lab=self.feat_lab,
            feat_dim_lab=self.feat_dim_lab,
            ids=self.ids,
            extra=self.extra,
        )

    def test_bare_constructor(self):
        ''' Test calling the constructor with only the required parameters. '''
        d = DataSet(self.data)
        np.testing.assert_equal(d.data, self.data)
        self.assertEqual(d.nfeatures, 5*3)
        self.assertEqual(d.feat_shape, (5,3))
        self.assertEqual(d.feat_lab, [range(5), range(3)])
        self.assertEqual(d.feat_dim_lab, ['feat_dim0', 'feat_dim1'])
        np.testing.assert_equal(d.labels, np.ones((1, 10), dtype=np.bool))
        self.assertEqual(d.nclasses, 1)
        self.assertEqual(d.cl_lab, ['class0'])
        np.testing.assert_equal(d.ids, np.atleast_2d(range(10)))
        self.assertEqual(d.extra, {})

    def test_full_constructor(self):
        ''' Test calling the constructor with all parameters '''
        d = self.d # The default DataSet used a full constructor

        np.testing.assert_equal(d.data, self.data)
        self.assertEqual(d.nfeatures, 5*3)
        self.assertEqual(d.feat_shape, (5,3))
        self.assertEqual(d.feat_lab, self.feat_lab)
        self.assertEqual(d.feat_dim_lab, self.feat_dim_lab)
        np.testing.assert_equal(d.labels, self.labels)
        self.assertEqual(d.nclasses, 5)
        self.assertEqual(d.cl_lab, self.cl_lab)
        np.testing.assert_equal(d.ids, self.ids)
        self.assertEqual(d.extra, self.extra)

    def test_default(self):
        ''' Test copying values from a default DataSet '''
        data = np.random.rand(5,3,10)
        d = DataSet(data=data, default=self.d)

        np.testing.assert_equal(d.data, data)
        self.assertEqual(d.nfeatures, 5*3)
        self.assertEqual(d.feat_shape, (5,3))
        self.assertEqual(d.feat_lab, self.feat_lab)
        self.assertEqual(d.feat_dim_lab, self.feat_dim_lab)
        np.testing.assert_equal(d.labels, self.labels)
        self.assertEqual(d.nclasses, 5)
        self.assertEqual(d.cl_lab, self.cl_lab)
        np.testing.assert_equal(d.ids, self.ids)
        self.assertEqual(d.extra, self.extra)

    def test_compatable_default(self):
        ''' Feature shape is different now, so some fields of the default
        DataSet are not applicable here. '''
        data = np.random.rand(4,3,10)
        d = DataSet(data=data, default=self.d)

        np.testing.assert_equal(d.data, data)
        self.assertEqual(d.nfeatures, 4*3)
        self.assertEqual(d.feat_shape, (4,3))
        self.assertEqual(d.feat_lab, [range(4), range(3)])
        self.assertEqual(d.feat_dim_lab, self.feat_dim_lab)
        np.testing.assert_equal(d.labels, self.labels)
        self.assertEqual(d.nclasses, 5)
        self.assertEqual(d.cl_lab, self.cl_lab)
        np.testing.assert_equal(d.ids, self.ids)
        self.assertEqual(d.extra, self.extra)

        # In this case, no fields of the default DataSet are compatable, except
        # the 'extra' field
        data = np.random.rand(5,7)
        d = DataSet(data=data, default=self.d)

        np.testing.assert_equal(d.data, data)
        self.assertEqual(d.nfeatures, 5)
        self.assertEqual(d.feat_shape, (5,))
        self.assertEqual(d.feat_lab, [range(5)])
        self.assertEqual(d.feat_dim_lab, ['feat_dim0'])
        np.testing.assert_equal(d.labels, np.ones((1, 7), dtype=np.bool))
        self.assertEqual(d.nclasses, 1)
        self.assertEqual(d.cl_lab, ['class0'])
        np.testing.assert_equal(d.ids, np.atleast_2d(range(7)))
        self.assertEqual(d.extra, self.extra)

    def test_construction_empty(self):
        '''Test empty construction of DataSet'''
        data = np.zeros((0, 0))
        labels = np.zeros((1, 0))
        d = DataSet(data, labels)
        self.assertEqual(d.ninstances, 0)
        self.assertEqual(d.ninstances_per_class, [0])
        self.assertEqual(d.nfeatures, 0)
        self.assertEqual(d.nclasses, 1)
        self.assertEqual(d.extra, {})

    def test_conversion(self):
        ''' Test whether inputs to the constructor are properly converted '''
        d = DataSet(0)
        np.testing.assert_equal(d.data, [[0]])

        d = DataSet([1.1, 2.2, 3.3])
        np.testing.assert_equal(d.data, [[1.1, 2.2, 3.3]])

        d = DataSet(self.data, labels=range(10))
        np.testing.assert_equal(d.labels, np.atleast_2d(range(10)))
        self.assertEqual(d.nclasses, 10)

        d = DataSet(self.data, ids=range(10))
        np.testing.assert_equal(d.ids, np.atleast_2d(range(10)))

    def test_X(self):
        ''' Test the 'X' property '''
        d = DataSet(range(10))
        self.assertEqual(d.X.ndim, 2)
        np.testing.assert_equal(d.X, np.atleast_2d(range(10)).T)

        d = self.d
        self.assertEqual(d.X.ndim, 2)
        np.testing.assert_equal(d.X, self.data.reshape(-1, d.ninstances).T)

    def test_y(self):
        ''' Test the 'y' property '''
        d = self.d
        self.assertEqual(d.y.ndim, 1)
        self.assertEqual(d.y.dtype, np.int)
        np.testing.assert_equal(d.y, self.labels.flat)

        labels = to_one_of_n(self.labels.flat)
        d = DataSet(labels=labels, default=self.d)
        self.assertEqual(d.y.ndim, 1)
        self.assertEqual(d.y.dtype, np.int)
        np.testing.assert_equal(d.y, self.labels.flat)

        labels = np.random.rand(5, 10)
        d = DataSet(labels=labels, default=self.d)
        self.assertEqual(d.y.ndim, 1)
        self.assertEqual(d.y.dtype, np.int)
        np.testing.assert_equal(d.y, np.argmax(labels, axis=0))

    def test_ninstances(self):
        ''' Test the 'ninstances' property '''
        d = self.d
        self.assertEqual(d.ninstances, 10)

        d = DataSet(range(243))
        self.assertEqual(d.ninstances, 243)

    def test_nclasses(self):
        ''' Test the 'nclasses' property '''
        d = self.d
        self.assertEqual(d.nclasses, 5)

        d = DataSet(labels=np.arange(10, dtype=np.int), default=d)
        self.assertEqual(d.nclasses, 10)

        d = DataSet(labels=np.arange(10, dtype=np.float), default=d)
        self.assertEqual(d.nclasses, 1)

        d = DataSet(labels=np.zeros((1, 10), dtype=np.bool), default=d)
        self.assertEqual(d.nclasses, 1)

        d = DataSet(labels=np.ones((1, 10), dtype=np.bool), default=d)
        self.assertEqual(d.nclasses, 1)

    def test_ninstances_per_class(self):
        ''' Test the 'ninstances_per_class' property '''
        d = self.d
        np.testing.assert_equal(d.ninstances_per_class, [2,2,2,2,2])

        d = DataSet(labels=np.arange(10, dtype=np.int), default=d)
        np.testing.assert_equal(d.ninstances_per_class, [1,1,1,1,1,1,1,1,1,1])

        d = DataSet(labels=np.arange(10, dtype=np.float), default=d)
        self.assertEqual(d.ninstances_per_class, [10])

        d = DataSet(labels=np.zeros((1, 10), dtype=np.bool), default=d)
        self.assertEqual(d.ninstances_per_class, [0])

        d = DataSet(labels=np.ones((1, 10), dtype=np.bool), default=d)
        self.assertEqual(d.ninstances_per_class, [10])

        d = DataSet(labels=np.eye(10, dtype=np.bool), default=d)
        np.testing.assert_equal(d.ninstances_per_class, [1,1,1,1,1,1,1,1,1,1])

        d = DataSet(labels=np.eye(10, dtype=np.float), default=d)
        np.testing.assert_equal(d.ninstances_per_class, [1,1,1,1,1,1,1,1,1,1])
        
        d = DataSet(labels=np.eye(10) + np.random.rand(10,10), default=d)
        np.testing.assert_equal(d.ninstances_per_class, [1,1,1,1,1,1,1,1,1,1])

    def test_feat_shape(self):
        ''' Test the 'feat_shape' property '''
        d = self.d
        self.assertEqual(d.feat_shape, (5,3))

        d = DataSet(range(10))
        self.assertEqual(d.feat_shape, (1,))

    def test_constructor_errors(self):
        ''' Test whether errors are raised for invalid constructor arguments '''
        self.assertRaises(ValueError, DataSet) # no data
        self.assertRaises(ValueError, DataSet, labels=[0]) # no data
    
        data = labels = ids = np.arange(12)
        
        # Test wrong types
        self.assertRaises(AssertionError, DataSet, data=data, labels=labels,
            ids=ids, cl_lab='c0')
        self.assertRaises(AssertionError, DataSet, data=data, labels=labels,
            ids=ids, feat_lab='f0')
        self.assertRaises(AssertionError, DataSet, data=data, labels=labels,
            ids=ids, extra='baz')
        self.assertRaises(AssertionError, DataSet, data=data, labels=labels, ids=ids, 
            feat_dim_lab='baz')

        # Test wrong sizes
        data = np.random.rand(5,4)
        self.assertRaises(ValueError, DataSet, data, labels=[0])
        self.assertRaises(ValueError, DataSet, data, ids=[0])
        self.assertRaises(ValueError, DataSet, data, feat_lab=['a'])
        self.assertRaises(ValueError, DataSet, data,
            feat_lab=[['a', 'b', 'c', 'd', 'e'], [1,2,3,4]])
        self.assertRaises(ValueError, DataSet, data,
            feat_dim_lab=['dim1', 'dim2'])
        self.assertRaises(ValueError, DataSet, data, labels=range(4),
            cl_lab=['class1'])


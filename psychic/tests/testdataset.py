from ..dataset import DataSet, concatenate, as_instances
from ..helpers import to_one_of_n
import numpy as np
import unittest
import tempfile

class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(5,3,10)
        self.labels = np.atleast_2d(7+np.arange(5, dtype=np.int).repeat(2))
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
        np.testing.assert_equal(d.possible_labels, range(7, 12))
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
        np.testing.assert_equal(d.possible_labels, self.d.possible_labels)
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
        np.testing.assert_equal(d.possible_labels, self.d.possible_labels)
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
        self.assertFalse(hasattr(d, 'possible_labels'))
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
        np.testing.assert_equal(d.y, self.labels.flat - np.min(self.labels))

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

        d_int = DataSet(labels=np.arange(10, dtype=np.int), default=d)
        self.assertEqual(d_int.nclasses, 10)

        d_float = DataSet(labels=np.eye(10, dtype=np.float), default=d)
        self.assertEqual(d_float.nclasses, 10)

        d_bool = DataSet(labels=np.zeros((5, 10), dtype=np.bool), default=d)
        self.assertEqual(d_bool.nclasses, 5)

        d = DataSet(labels=np.ones((1, 10), dtype=np.bool), default=d)
        self.assertEqual(d.nclasses, 1)

        # When dealing with integer labels, the number of defined classes should
        # not change when slicing a part of the dataset
        self.assertEquals(d_int[:2].nclasses, 10)
        self.assertEquals(d_bool[:2].nclasses, 5)
        self.assertEquals(d_float[:2].nclasses, 10)

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
        self.assertRaises(ValueError, DataSet, data, labels=[0])
        self.assertRaises(ValueError, DataSet, data, ids=[0])
        self.assertRaises(ValueError, DataSet, data, feat_lab=['a'])
        self.assertRaises(ValueError, DataSet, data,
            feat_lab=[['a', 'b', 'c', 'd', 'e'], [1,2,3,4]])
        self.assertRaises(ValueError, DataSet, data,
            feat_dim_lab=['dim1', 'dim2'])
        self.assertRaises(ValueError, DataSet, data, labels=range(4),
            cl_lab=['class1'])

        # Test wrong dimensions
        self.assertRaises(ValueError, DataSet, data, labels=np.zeros((1,1,12)))
        self.assertRaises(ValueError, DataSet, data, ids=np.zeros((1,1,12)))

        # Test duplicate ids
        self.assertRaises(ValueError, DataSet, data, ids=np.ones(12))

        # Test wrong unique_labels
        self.assertRaises(ValueError, DataSet, data, labels=labels,
            possible_labels=[1,2])

    def test_finite_feats(self):
        labels = np.ones((2, 10))
        for v in [np.inf, -np.inf, np.nan]:
            data = np.zeros((2, 10))
            DataSet(data.copy(), labels) # no error
            data[1, 5] = v
            self.assertRaises(ValueError, DataSet, data, labels)

    def test_get_class(self):
        # Integer labels
        d_int = DataSet(labels=np.arange(5, dtype=np.int).repeat(2),
                    default=self.d)

        # Boolean labels
        d_bool = DataSet(labels=np.eye(5, dtype=np.bool).repeat(2, axis=1),
                    default=self.d)

        # Float labels
        d_float = DataSet(labels=np.eye(5, dtype=np.float).repeat(2, axis=1),
                    default=self.d)

        for i,lab in enumerate(self.cl_lab):
            instances = [i*2,i*2+1]
            self.assertEqual(d_int.get_class(i), d_int[instances])
            self.assertEqual(d_int.get_class(lab), d_int[instances])
            self.assertEqual(d_bool.get_class(i), d_bool[instances])
            self.assertEqual(d_bool.get_class(lab), d_bool[instances])
            self.assertEqual(d_float.get_class(i), d_float[instances])
            self.assertEqual(d_float.get_class(lab), d_float[instances])

        self.assertRaises(ValueError, d_int.get_class, {'a':'b'})

    def test_prior(self):
        np.testing.assert_equal(self.d.prior, [.2, .2, .2, .2, .2])
        np.testing.assert_equal(DataSet([0]).prior, [1])
        np.testing.assert_equal(DataSet([0], [[True], [False]]).prior, [1, 0])
        np.testing.assert_equal(DataSet([0], [[1.], [0.]]).prior, [1, 0])

    def test_shuffle(self):
        '''Test shuffling the DataSet'''
        data, labels = np.random.random((2, 6)), np.ones(6) 
        d = DataSet(data, labels)

        # shuffle and sort
        ds = d.shuffled()
        self.failIfEqual(ds, d)
        self.assertEqual(ds[np.argsort(ds.ids.flat)], d)

    def test_sorted(self):
        '''Test sorting the DataSet'''
        ids = np.array([[0, 1, 2, 3, 4, 5], [1, 1, 1, 0, 0, 0]])
        data, labels = np.random.random((2, 6)), np.ones(6) 
        d1d = DataSet(data, labels, ids[0])
        d2d = DataSet(data, labels, ids)

        # shuffle and sort
        d1ds = d1d.shuffled()
        self.failIfEqual(d1d, d1ds)
        self.assertEqual(d1d, d1ds.sorted())

        d2ds = d2d.shuffled()
        self.failIfEqual(d2d, d2ds)
        self.assertEqual(d2d, d2ds.sorted())

    def test_equality(self):
        d = self.d
        diff_ds = [DataSet(data=d.data+1, default=d),
            DataSet(labels=d.labels+1, default=d),
            DataSet(ids=d.ids+1, default=d),
            DataSet(cl_lab=['a', 'b', 'c', 'd', 'e'], default=d),
            DataSet(feat_lab=[['F1', 'F2', 'F3', 'F4', 'F5'],
                              [0, 1, 2]], default=d),
            DataSet(feat_dim_lab=['da', 'db'], default=d),
            DataSet(extra={'foo':'baz'}, default=d),
            d[:0]]

        self.assertEqual(d, d)
        self.assertEqual(d, DataSet(data=d.data, labels=d.labels, ids=d.ids,
            feat_lab=d.feat_lab, cl_lab=d.cl_lab, feat_dim_lab=d.feat_dim_lab,
            extra=d.extra))

        # test all kinds of differences
        for dd in diff_ds:
            self.failIfEqual(dd, d)
        
        # test special cases
        self.assertEqual(d, DataSet(data=d.data.copy(), labels=d.labels.copy(),
            ids=d.ids.copy(), default=d))
        self.failIfEqual(d, 3)
        self.failIfEqual(d[:0], d) # triggered special cast in np.array comparison.
        self.failIfEqual(d[:0], d[0]) # similar

    def test_add(self):
        '''Test the creation of compound datasets using the add-operator.'''
        ids = np.array([[0, 1, 2, 3, 4, 5], [1, 1, 1, 0, 0, 0]])
        data, labels = np.random.random((3, 6)), np.ones((3, 6)) 
        d = DataSet(data=data, labels=labels, ids=ids,
                    feat_lab=['feat%d' for d in range(3)])

        da, db = d[:3], d[3:]
        self.assertEqual(da + db, d)

        # different nfeatures
        self.assertRaises(ValueError, da.__add__,
            DataSet(data=db.data[:-1], feat_lab=d.feat_lab[0][:-1], default=db))
        
        # different nclasses
        self.assertRaises(ValueError, da.__add__,
            DataSet(labels=db.labels[:-1], cl_lab=d.cl_lab[:-1], default=db))

        # different feat_lab
        self.assertRaises(ValueError, da.__add__,
            DataSet(feat_lab=[['f0', 'f1', 'f2']], default=db))

        # different feat_shape
        self.assertRaises(ValueError, da.__add__,
            DataSet(data=np.random.random((1,3,6)), default=db))
        
        # different cl_lab
        self.assertRaises(ValueError, da.__add__,
            DataSet(cl_lab=['c0', 'c1', 'c2'], default=db))

        # different feat_dim_lab
        self.assertRaises(ValueError, da.__add__,
            DataSet(feat_dim_lab=['cm'], default=db))

        # different extra
        self.assertRaises(ValueError, da.__add__,
            DataSet(extra={'foo':'baz'}, default=db))

        # add single instances together
        self.assertEqual(reduce(lambda a,b: a + b, [d[i] for i in range(6)]), d)

        # add empty datasets
        self.assertEqual(d+d[:0], d)
        self.assertEqual(d[:0]+d, d)
        self.assertRaises(AssertionError, d.append, 'foo')

    def test_append(self):
        '''Test appending DataSets.'''
        d = self.d
        d2 = DataSet(ids=1 + np.arange(d.ninstances)/10., default=d)
        self.assertEqual(d.append(d2), d+d2)
        self.assertEqual(d.append(d, ignore_index=True).ninstances,
            2*d.ninstances)
        self.assertRaises(ValueError, d.append, d)

    def test_concatenate(self):
        '''Test concatenating multiple DataSets.'''

        d = self.d
        d2 = DataSet(ids=1 + np.arange(d.ninstances)/10., default=d)
        d3 = DataSet(ids=2 + np.arange(d.ninstances)/10., default=d)

        self.assertEqual(concatenate([d]), d)
        self.assertEqual(concatenate([d, d2, d3]), d+d2+d3)
        self.assertEqual(concatenate([d,d,d], ignore_index=True).ninstances,
            3*d.ninstances)
        self.assertRaises(ValueError, concatenate, [d,d,d])

        d = DataSet(labels=np.eye(5).repeat(2, axis=1), default=d)
        self.assertEqual(concatenate([d, d[:0]]), d)

        self.assertRaises(ValueError, concatenate, [d,'foo'])
        self.assertRaises(ValueError, concatenate, [d,DataSet([0])])

        # Datasets with integer labels
        

    def test_indexing(self):
        '''Test the indexing of DataSet.'''
        d = DataSet(labels=np.eye(5).repeat(2, axis=1), default=self.d)

        # check if all members are correctly extracted
        d0 = DataSet(data=d.data[:,:,[0]], labels=d.labels[:,[0]],
                ids=d.ids[:,[0]], default=d)
        self.assertEqual(d[0], d0)

        # test high-level properties
        self.assertEqual(d[:], d)
        self.assertEqual(reduce(lambda a,b: a + b, [d[i] for i in range(10)]), d)
        self.assertEqual(d[:-9], d[0])

        # test various indexing types
        indices = np.arange(d.ninstances)
        self.assertEqual(d[indices==0], d[0])
        self.assertEqual(d[indices.tolist()], d)
        self.assertEqual(d[[1]], d[1])

        # invalid indexing type
        self.assertRaises(ValueError, d.__getitem__, 'foo')

        # when dealing with integer labels, the number of classes should not
        # change due to slicing. 
        self.assertEqual(self.d[[2,3]].nclasses, 5)
        self.assertEqual(self.d[[2,3]].cl_lab, self.cl_lab)

    def test_ix_indexing(self):
        ''' Test indexing of DataSet using the .ix property.'''
        d = self.d
        d_sliced = DataSet(
            data=d.data[:2, :2, :2],
            labels=d.labels[:, :2],
            ids=d.ids[:,:2],
            feat_lab=[f[:2] for f in d.feat_lab[:2]],
            default=d
        )
        self.assertEqual(d.ix[:,:,:], d)

        # Slice on all axes, using different indexing types
        self.assertEqual(d.ix[:2,:2,:2], d_sliced)
        self.assertEqual(d.ix[[0,1],[0,1],[0,1]], d_sliced)
        self.assertEqual(d.ix[:-3,:-1,:-8], d_sliced)
        self.assertEqual(d.ix[:2], d[:2])

        # when dealing with integer labels, the number of classes should not
        # change due to slicing. 
        self.assertEqual(d.ix[:,:,[2,3]].nclasses, 5)

        # Test dropping in dimensionality
        self.assertEqual(d.ix[2,:,:].data.shape, (3, 10))
        self.assertEqual(d.ix[[2],:,:].data.shape, (1, 3, 10))
        
        # Never drop the last axis (=instances)
        self.assertEqual(d.ix[:,:,[2]].data.shape, (5, 3, 1))
        self.assertEqual(d.ix[:,:,2].data.shape, (5, 3, 1))

    def test_lix_indexing(self):
        ''' Test indexing of DataSet using the .lix property.'''
        d = self.d
        d_sliced = DataSet(
            data=d.data[:2, :2, :2],
            labels=d.labels[:, :2],
            ids=d.ids[:,:2],
            feat_lab=[f[:2] for f in d.feat_lab[:2]],
            default=d
        )
        self.assertEqual(d.ix[:,:,:], d)

        # Slice on all axes, using different indexing types
        self.assertEqual(d.lix[:'b',:3,:.2], d_sliced)
        self.assertEqual(d.lix['a':'b',1:3,0:.2], d_sliced)
        self.assertEqual(d.lix[['a','b'],[1,2],[0.0,0.1]], d_sliced)

        # Test dropping in dimensionality
        self.assertEqual(d.lix['c',:,:].data.shape, (3, 10))
        self.assertEqual(d.lix[['c'],:,:].data.shape, (1, 3, 10))
        
        # Never drop the last axis (=instances)
        self.assertEqual(d.lix[:,:,[0.2]].data.shape, (5, 3, 1))
        self.assertEqual(d.lix[:,:,0.2].data.shape, (5, 3, 1))

        # Test indexing with floats
        self.assertEqual(d.lix[:,:,:0.125].data.shape, (5, 3, 2))
        self.assertEqual(d.lix[:,1.5:2.5,0.125:].data.shape, (5, 1, 8))

        # One second of data at 100Hz, should be exactly 100 samples
        d = DataSet(np.random.rand(1000), ids=np.arange(1000)/100.)
        self.assertEqual(d.lix[:,4:5].ninstances, 100)

    def test_str(self):
        '''Test string representation.'''
        self.assertEqual(
            "DataSet with 10 instances, 15 features [5x3], 5 classes: [2, 2, "+
            "2, 2, 2], extras: ['foo']", str(self.d))

    def test_repr(self):
        '''Test string representation.'''
        self.assertEqual(
            "DataSet with 10 instances, 15 features [5x3], 5 classes: [2, 2, "+
            "2, 2, 2], extras: ['foo']", str(self.d))

    def test_save_load(self):
        '''Test loading and saving datasets'''
        # test round-trip using file objects
        _, tfname = tempfile.mkstemp('.dat')
        self.d.save(open(tfname, 'wb'))
        self.assertEqual(self.d, DataSet.load(open(tfname, 'rb')))

        # test round-trip using filenames
        _, tfname = tempfile.mkstemp('.dat')
        self.d.save(tfname)
        self.assertEqual(self.d, DataSet.load(tfname))

    def test_write_protected(self):
        d = self.d
        for att in [d.data, d.labels, d.ids]:
            self.assertRaises(ValueError, att.__setitem__, (0, 0), -1)

    def test_as_instances(self):
        ds = [DataSet(np.random.rand(5,4,1)) for _ in range(10)]
        self.assertEqual(as_instances(ds).data.shape, (5,4,1,10))

        labels = np.arange(5).repeat(2)
        ids = np.atleast_2d(np.arange(10) / 10.)
        self.assertEqual(as_instances(ds, labels).nclasses, 5)
        self.assertEqual(as_instances(ds, to_one_of_n(labels)).nclasses, 5)
        np.testing.assert_equal(as_instances(ds, ids=ids).ids, ids)

        # Argument must be a nonempty list
        self.assertRaises(AssertionError, as_instances, [])

        # Wrong number of features
        ds[5] = DataSet(np.random.rand(5,4,2))
        self.assertRaises(AssertionError, as_instances, ds)

class TestEmpty(unittest.TestCase):
    def setUp(self):
        self.d0 = DataSet(data=np.zeros((10, 0)), labels=np.zeros((3, 0)))

    def test_props(self):
        d0 = self.d0
        self.assertEqual(d0.ninstances, 0)
        self.assertEqual(d0.nfeatures, 10)
        self.assertEqual(d0.nclasses, 3)
        self.assertEqual(d0.ninstances_per_class, [0, 0, 0])

    def test_sort(self): 
        d0 = self.d0
        ds = d0.sorted()
        self.assertEqual(ds, d0)
        self.assertNotEqual(id(ds), id(d0))

    def test_shuffle(self): 
        d0 = self.d0
        ds = d0.shuffled()
        self.assertEqual(ds, d0)
        self.assertNotEqual(id(ds), id(d0))

    def test_bounds(self):
        d0 = self.d0
        self.assertRaises(IndexError, d0.__getitem__, 0)
        self.assertRaises(IndexError, d0.__getitem__, 1)
        self.assertEqual(d0[:], d0)
        self.assertEqual(d0[[]], d0)
        self.assertEqual(d0[np.asarray([])], d0)

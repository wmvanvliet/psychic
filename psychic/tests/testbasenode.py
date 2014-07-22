import unittest
import numpy as np
from .. import DataSet
from ..nodes import BaseNode

class TestBaseNode(unittest.TestCase):
    def setUp(self):
        data = np.random.rand(4, 10)
        labels = np.ones((2, 10))
        self.d = DataSet(data, labels)
        self.n = BaseNode()

    def test_existing_methods(self):
        '''Test that masking an exisiting method raises an exception'''
        class MaskTrain(BaseNode):
            def train(self, d):
                pass

        class MaskApply(BaseNode):
            def apply(self, d):
                pass
        self.assertRaises(Exception, MaskTrain)
        self.assertRaises(Exception, MaskApply)

    def test_compatible_train_test(self):
        d = self.d
        n = self.n
        n.train(d)
        n.apply(d) # no exception

        n.train(d)
        self.assertRaises(ValueError, n.apply,
            DataSet(data=self.d.data.reshape(2, 2, -1), default=d))

    def test_logger_name(self):
        class TestNode(BaseNode):
            pass

        n = BaseNode() 
        self.assertEqual(n.log.name, 'psychic.nodes.BaseNode')

        tn = TestNode()
        self.assertEqual(tn.log.name, 'psychic.nodes.TestNode')

    def test_train_apply(self):
        class TestNode(BaseNode):
            def train_(self, d):
                self.a = 5

            def apply_(self, d):
                return DataSet(data=d.data * self.a, default=d)

        n = TestNode()
        d = self.d
        self.assertEqual(n.train(d).apply(d), n.train_apply(d))
        self.assertEqual(n.train_apply(d), n.train_apply(d,d))

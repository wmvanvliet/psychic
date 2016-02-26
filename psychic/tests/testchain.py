import unittest
import numpy as np
from ..nodes import BaseNode, Chain
from .. import DataSet

try:
    from sklearn import svm
    from sklearn import linear_model
    from sklearn import preprocessing
    sklearn_present = True
except:
    sklearn_present = False

class NOPNode(BaseNode):
    def train_(self, d):
        self.d = d

    def apply_(self, d):
        return d

class AddSumNode(BaseNode):
    def __init__(self):
        BaseNode.__init__(self)
        self.train_calls = 0
        self.test_calls = 0
        self.sum = None

    def train_(self, d):
        self.sum = np.sum(d.data)
        self.train_calls += 1

    def apply_(self, d):
        self.test_calls += 1
        return DataSet(data=d.data + self.sum, default=d)

class TestChain(unittest.TestCase):
    def setUp(self):
        self.d = DataSet(data=np.ones((1, 10)), labels=np.random.rand(2, 10))
        self.nodes = [AddSumNode() for n in range(3)]
        self.c = Chain(self.nodes)

    def test_chain(self):
        d = self.d
        self.c.train(d)
        np.testing.assert_equal([n.train_calls for n in self.nodes], [1, 1, 1])
        np.testing.assert_equal([n.test_calls for n in self.nodes], [1, 1, 0])
        np.testing.assert_equal([n.sum for n in self.nodes], 
            [10, (1 + 10) * 10, (1 + 10 + 110) * 10])

        np.testing.assert_equal(self.c.apply(d).data, 
            1 + 10 + (1 + 10) * 10 + (1 + 10 + (1 + 10) * 10) * 10 * d.data)

        np.testing.assert_equal([n.test_calls for n in self.nodes], [2, 2, 1])

    def test_train_apply(self):
        d = self.d
        self.c.train_apply(d)
        np.testing.assert_equal([n.train_calls for n in self.nodes], [1, 1, 1])
        np.testing.assert_equal([n.test_calls for n in self.nodes], [1, 1, 1])

    def test_train_sklearn(self):
        if not sklearn_present:
            return

        ch = Chain([NOPNode(), svm.LinearSVC()])
        ch.train(self.d)
        self.assertTrue(hasattr(ch.nodes[1], 'coef_'))

        ch = Chain([svm.LinearSVC(), NOPNode()])
        ch.train(self.d)
        self.assertTrue(hasattr(ch.nodes[0], 'coef_'))
        self.assertEqual(ch.nodes[1].d.data.shape, (1, 10))

    def test_apply_sklearn(self):
        if not sklearn_present:
            return

        labels = np.zeros((2,10))
        labels[0,:5] = 1
        labels[1,5:] = 1
        d = DataSet(np.ones((5,10,5)), labels=labels[:,:5])
        d += DataSet(np.zeros((5,10,5)), labels=labels[:,5:],
                     ids=np.arange(5,10))

        # Node that predicts integers (SVM)
        ch = Chain([NOPNode(), svm.LinearSVC()])
        d2 = ch.train_apply(d)
        np.testing.assert_equal(d2.data, d2.labels)

        # Node that predicts floats (OLS)
        ch = Chain([NOPNode(), linear_model.LinearRegression()])
        d2 = ch.train_apply(d)
        self.assertEqual(d2.data.shape, (1, 10))

        # Node that predicts probabilities
        ch = Chain([NOPNode(), linear_model.LogisticRegression()])
        d2 = ch.train_apply(d)
        self.assertEqual(d2.data.shape, (2, 10))
        self.assertTrue(np.all(d2.data > 0))
        self.assertTrue(np.all(d2.data < 1))
        np.testing.assert_equal(d2.y, np.r_[np.zeros(5), np.ones(5)])

        # Node that only implements a transform
        ch = Chain([NOPNode(), preprocessing.StandardScaler()])
        d2 = ch.train_apply(d)
        np.testing.assert_equal(np.mean(d2.data, axis=1), 0)
        np.testing.assert_equal(np.std(d2.data, axis=1), 1.)

        # When node is not the last node, transform function should be applied,
        ch = Chain([preprocessing.StandardScaler(), NOPNode()])
        self.assertEqual(ch.train_apply(d), d2)
        
        # When node is not the last node, transform function should be applied,
        # but LinearRegression does not have a transform. Predict function
        # should be called in this case.
        ch = Chain([linear_model.LinearRegression()])
        d2 = ch.train_apply(d)
        ch = Chain([linear_model.LinearRegression(), NOPNode()])
        self.assertEqual(ch.train_apply(d), d2)

    def test_str(self):
        ch = Chain([AddSumNode(), NOPNode()])
        self.assertEqual(str(ch), 'Chain (AddSumNode ->\nNOPNode)')

        if sklearn_present:
            ch = Chain([linear_model.LinearRegression()])
            self.assertEqual(str(ch),
                'Chain (LinearRegression(copy_X=True, fit_intercept=True, '+
                'n_jobs=1, normalize=False))')

import numpy as np
from basenode import BaseNode
from ..dataset import DataSet
from ..helpers import to_one_of_n

def _apply_sklearn(n, d, last_node=False):
    if n.__module__.startswith('sklearn'):
        # Use the most suitable function
        if not last_node and hasattr(n, 'transform'):
            print 'transform'
            X = n.transform(d.X)
        elif hasattr(n, 'predict_proba'):
            print 'predict_proba'
            X = n.predict_proba(d.X)
        elif hasattr(n, 'predict'):
            print 'predict'
            p = n.predict(d.X)
            if p.dtype == np.float:
                X = p
            else:
                X = to_one_of_n(p.T, range(d.nclasses)).T
        elif last_node and hasattr(n, 'transform'):
            print 'transform'
            X = n.transform(d.X)
        else:
            raise ValueError(
                'node' + repr(n) + ' doesn\'t have a transform, '+
                'predict_proba or predict function')

        return DataSet(data=X.T, default=d)
    else:
        return n.apply(d)

class Chain(BaseNode):
    def __init__(self, nodes):
        BaseNode.__init__(self)
        self.nodes = list(nodes)
            
    def _pre_process(self, d): 
        ''' Train and apply all nodes but the last '''
        for n in self.nodes[:-1]:
            self.log.info('Processing with %s...' % str(n))
            self.log.debug('d = %s' % d)

            # Check whether this node comes from scikit-learn
            if n.__module__.startswith('sklearn'):
                if hasattr(n, 'fit_transform'):
                    X = n.fit_transform(d.X, d.y)
                elif hasattr(n, 'transform'):
                    n.fit(d.X, d.y)
                    X = n.transform(d.X)
                elif hasattr(n, 'predict_proba'):
                    n.fit(d.X, d.y)
                    X = n.predict_proba(d.X)
                elif hasattr(n, 'predict'):
                    n.fit(d.X, d.y)
                    p = n.predict(d.X)
                    if p.dtype == np.float:
                        X = p
                    else:
                        X = to_one_of_n(p.T, range(d.nclasses)).T
                d = DataSet(data=X.T, default=d)
            else:
                d = n.train(d).apply(d)
        
        return d


    def train_(self, d):
        d = self._pre_process(d)
        n = self.nodes[-1]

        self.log.info('Training %s...' % str(n))
        self.log.debug('d = %s' % d)

        if n.__module__.startswith('sklearn'):
            n.fit(d.X, d.y)
        else:
            n.train(d)

    def apply_(self, d): 
        for i,n in enumerate(self.nodes):
                self.log.info('Testing with %s...' % str(n))
                self.log.debug('d = %s' % d)
                d = _apply_sklearn(n, d, last_node = (i==len(self.nodes)-1))

        return d

    def train_apply(self, dtrain, dtest=None):
        if dtest != None:
            return self.train(dtrain).apply(dtest)
        else:
            # Train and test set are the same, do some optimization
            dtrain = self._pre_process(dtrain)

            n = self.nodes[-1]
            if n.__module__.startswith('sklearn'):
                n.fit(dtrain.X, dtrain.y)
            else:
                n.train(dtrain)
            return _apply_sklearn(n, dtrain, last_node=True)

    def __str__(self):
        return 'Chain (%s)' % ' ->\n'.join([str(n) for n in self.nodes])

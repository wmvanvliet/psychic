import logging, warnings

class BaseNode:
    def __init__(self):
        self.name = self.__class__.__name__
        self.empty_d = None

        # test for overridden train and test methods:
        if self.__class__.train != BaseNode.train:
            raise Exception(
                'Do not override methode train(). Use train_() instead.')
        if self.__class__.apply != BaseNode.apply:
            raise Exception(
                'Do not override methode apply(). Use apply_() instead.')

    @property
    def log(self):
        '''
        Logs are not deepcopy-able, therefore we need a property...
        '''
        return logging.getLogger('psychic.nodes.' + self.name)

    @property
    def nclasses(self):
        return self.empty_d.nclasses

    def test(self, d):
        warnings.warn('Method [Node].test() is deprecated, ' + 
            'use [Node].apply() instead.',
            DeprecationWarning)
        return self.apply(d)

    def train(self, d):
        self.log.info('training on ' + str(d))

        # store format of d
        self.empty_d = d[:0]

        # delegate call
        self.train_(d)

        return self

    def apply(self, d):
        self.log.info('testing on ' + str(d))

        # check validity of d
        if self.empty_d != None and self.empty_d.feat_shape != d.feat_shape:
            raise ValueError('Node was trained with differently shaped data. '+
                '(was %s, is now %s)' % (self.empty_d.feat_shape, d.feat_shape))

        # delegate call
        return self.apply_(d)

    def train_apply(self, dtrain, dtest=None):
        '''
        Convenience method to train the node on dtrain, and apply it to dtest.
        If dtest is not given, dtrain is also used as dtest.
        '''
        self.train(dtrain)

        if dtest != None:
            return self.apply(dtest)
        else:
            return self.apply(dtrain)

    def __str__(self):
        return self.name

    def train_(self, d):
        '''
        Placeholder, meant to be replaced with the derived nodes train method.
        '''

    def apply_(self, d):
        '''
        Placeholder, meant to be replaced with the derived nodes apply method.
        '''
        pass

    def test_(self, d):
        '''Deprecated '''
        pass

'''
@package keraflow.regularizers
Built-in regularizers.
'''
from . import backend as B


class L1(object):
    '''L1 weight regularization penalty, also known as LASSO.
    '''
    def __init__(self, l1=0.01):
        self.l1 = l1

    def __call__(self, param):
        return B.sum(B.abs(param)) * B.cast_to_floatx(self.l1)


class L2(object):
    '''L2 weight regularization penalty, also known as weight decay, or Ridge.
    '''
    def __init__(self, l2=0.01):
        self.l2 = l2

    def __call__(self, param):
        return B.sum(B.square(param)) * B.cast_to_floatx(self.l2)


class L1L2(object):
    '''L1-L2 weight regularization penalty, also known as ElasticNet.
    '''
    def __init__(self, l1=0.01, l2=0.01):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, param):
        return B.sum(B.abs(param)) * B.cast_to_floatx(self.l1) + B.sum(B.square(param)) * B.cast_to_floatx(self.l2)

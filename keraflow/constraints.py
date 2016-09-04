'''
@package keraflow.constraints
Built-in constrints.
'''
from . import backend as B


class MaxNorm(object):
    '''Constrain the weights along an axis pattern to have unit norm.
    '''
    def __init__(self, m=2, axis=0):
        '''
        @param m: float. The target max norm.
        @param axis: int/list of int, axis pattern along which to calculate weight norms. Examples:
        1. In `Dense` layer, `W` has has shape (input_dim, output_dim), set `axis` to `0` to constrain each row vector of size (input_dim,).
        2. In a `Convolution2D`, `W` has shape (input_depth, output_depth, kernel_rows, kernel_cols), set `axis` to `[0,2,3]` to constrain the weights of each kernel tensor of size (input_depth, kernel_rows, kernel_cols).
        '''
        self.m = m
        self.axis = axis

    def __call__(self, param):
        norms = B.sqrt(B.sum(B.square(param), axis=self.axis, keepdims=True))
        clipped_norms = B.clip(norms, 0, self.m)
        param = param * (clipped_norms / (B.epsilon() + norms))
        return param


class NonNeg(object):
    '''Constrain the weights to be non-negative.
    '''
    def __call__(self, param):
        param *= B.cast(param >= 0., B.floatx())
        return param


class UnitNorm(object):
    '''Constrain the weights along an axis pattern to have unit norm.
    '''
    def __init__(self, axis=0):
        '''
        @param axis: int/list of int, axis pattern along which to calculate weight norms. Examples:
        1. In `Dense` layer, `W` has has shape (input_dim, output_dim), set `axis` to `0` to constrain each row vector of size (input_dim,).
        2. In a `Convolution2D`, `W` has shape (input_depth, output_depth, kernel_rows, kernel_cols), set `axis` to `[0,2,3]` to constrain the weights of each kernel tensor of size (input_depth, kernel_rows, kernel_cols).
        '''
        self.axis = axis

    def __call__(self, param):
        return param / (B.epsilon() + B.sqrt(B.sum(B.square(param), axis=self.axis, keepdims=True)))

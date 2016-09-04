'''
@package keraflow.objectives
Built-in objectives  functions.
All objectives should accept two 2D (or more) tensors `y_pred` & `y_true` and return the resulting 2D (or more) tensor containg objective score per sample.
'''
import numpy as np

from . import backend as B


def accuracy(y_pred, y_true):
    '''Accuracy objective. Note that this objective is a switch of binary_accuracy and categorical_accuracy.
    '''
    # TODO check efficiency of branching
    return B.switch(B.equal(B.shape(y_pred)[1], 1),
                    binary_accuracy(y_pred, y_true),
                    categorical_accuracy(y_pred, y_true))


def binary_accuracy(y_pred, y_true):
    return B.equal(B.round(y_pred), y_true)


def categorical_accuracy(y_pred, y_true):
    res = B.equal(B.argmax(y_pred, axis=1), B.argmax(y_true, axis=1))
    # res is 1D, but we must return 2D tensor
    return B.expand_dims(res, -1)


def square_error(y_pred, y_true):
    '''Squared error loss.
    '''
    return B.square(y_pred - y_true)


def absolute_error(y_pred, y_true):
    '''Absolute error loss.
    '''
    return B.abs(y_pred - y_true)


def absolute_percentage_error(y_pred, y_true):
    '''Absolute percentage error loss.
    '''
    return 100.* B.abs((y_pred - y_true) / B.clip(B.abs(y_true), B.epsilon(), np.inf))


def squared_logarithmic_error(y_pred, y_true):
    '''Squared log error loss.
    '''
    first_log = B.log(B.clip(y_pred, B.epsilon(), np.inf) + 1.)
    second_log = B.log(B.clip(y_true, B.epsilon(), np.inf) + 1.)
    return B.square(first_log - second_log)


def hinge(y_pred, y_true):
    '''Squared hinge error loss.
    '''
    return B.maximum(1. - y_pred * y_true, 0.)


def squared_hinge(y_pred, y_true):
    '''Squared hinge error loss.
    '''
    return B.square(B.maximum(1. - y_pred * y_true, 0.))


def binary_crossentropy(y_pred, y_true):
    ''' Crossentropy loss for binary labels. Shape of y_pred and y_true should be (N,1), where N is the sample size.
    '''
    y_true = B.clip(y_true, B.epsilon(), 1.-B.epsilon())
    y_pred = B.clip(y_pred, B.epsilon(), 1.-B.epsilon())
    return -(y_true * B.log(y_pred) + (1.0 - y_true) * B.log(1.0 - y_pred))


def categorical_crossentropy(y_pred, y_true):
    ''' Crossentropy loss for multiple-class classification. y_pred and y_true should be binary matrix of shape (N,c), where N is the sample size, c is the number of class.
    '''
    y_pred = B.clip(y_pred, B.epsilon(), 1.-B.epsilon())
    y_true = B.clip(y_true, B.epsilon(), 1)
    return -B.sum(y_true * B.log(y_pred), axis=B.ndim(y_pred)-1, keepdims=True)


def kullback_leibler_divergence(y_pred, y_true):
    '''Information gain from a predicted probability distribution Q to a true probability distribution P. Gives a measure of difference between both distributions.
    '''
    y_pred = B.clip(y_pred, B.epsilon(), 1)
    y_true = B.clip(y_true, B.epsilon(), 1)
    return B.sum(y_true * B.log(y_true / y_pred), axis=-1, keepdims=True)


def poisson(y_pred, y_true):
    '''Poisson loss.
    '''
    return y_true- y_pred * B.log(y_true+ B.epsilon())


def cosine_proximity(y_pred, y_true):
    '''Cosine proximity loss.
    '''

    def l2_normalize(x, axis):
        norm = B.sqrt(B.sum(B.square(x), axis=axis, keepdims=True))
        return x / norm

    y_pred = l2_normalize(y_pred, axis=-1)
    y_true= l2_normalize(y_true, axis=-1)
    return -y_pred * y_true

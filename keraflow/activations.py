'''
@package keraflow.activations
Built-in activation functions.
'''
from . import backend as B
from .utils import KeraFlowError as KError


def linear(x):
    ''' Linear activations.
    '''
    return x


def sigmoid(x):
    ''' Sigmoid activations.
    '''
    return B.sigmoid(x)


def hard_sigmoid(x):
    ''' hard_sigmoid activations.
    '''
    return B.hard_sigmoid(x)


def tanh(x):
    ''' tanh activations.
    '''
    return B.tanh(x)


def relu(x, alpha=0., max_value=None):
    ''' relu activations.
    '''
    return B.relu(x, alpha=alpha, max_value=max_value)


def softmax(x):
    ''' softmax activations. Applied across input tensor's last dimension.
    '''
    ndim = B.ndim(x)
    if ndim == 2:
        return B.softmax(x)
    elif ndim == 3:
        e = B.exp(x - B.max(x, axis=-1, keepdims=True))
        s = B.sum(e, axis=-1, keepdims=True)
        return e / s
    else:
        raise KError('Cannot apply softmax to a tensor of {}D (accepts only 2D or 3D).'.format(ndim))


def softplus(x):
    ''' Softplus activations.
    '''
    return B.softplus(x)


def softsign(x):
    ''' Softsign activations.
    '''
    return B.softsign(x)

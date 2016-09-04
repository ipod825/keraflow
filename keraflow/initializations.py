'''
@package keraflow.initializations
Built-in initialization functions.
'''
import numpy as np

from . import backend as B


def get_fans(shape):
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # assuming convolution kernels (2D or 3D).
        # kernel shape (input_depth, output_depth, k_rows, k_cols ...)
        receptive_field_size = np.prod(shape[2:])
        fan_in = shape[0]*receptive_field_size
        fan_out = shape[1]*receptive_field_size
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


def uniform(shape, scale=0.05, name=None):
    '''Unifrom initialization.
    '''
    return B.variable(np.random.uniform(low=-scale, high=scale, size=shape), name=name)


def normal(shape, scale=0.05, name=None):
    '''Normal initialization.
    '''
    return B.variable(np.random.normal(loc=0.0, scale=scale, size=shape), name=name)


def lecun_uniform(shape, name=None):
    '''Uniform initialization scaled by the square root of the number of inputs (LeCun 98).
    - Reference: [LeCun 98, Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    '''
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(3. / fan_in)
    return uniform(shape, scale, name=name)


def glorot_normal(shape, name=None):
    '''Gaussian initialization scaled by fan_in + fan_out.
    - Reference: Glorot & Bengio, AISTATS 2010
    '''
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s, name=name)


def glorot_uniform(shape, name=None):
    '''Uniform initialization scaled by fan_in + fan_out.
    '''
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s, name=name)


def he_normal(shape, name=None):
    ''' Gaussian initialization scaled by fan_in.
    - Reference: [He et al.,](http://arxiv.org/abs/1502.01852)
    '''
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / fan_in)
    return normal(shape, s, name=name)


def he_uniform(shape, name=None):
    '''he_uniform initialization.
    '''
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / fan_in)
    return uniform(shape, s, name=name)


def orthogonal(shape, scale=1.1, name=None):
    '''Use with square 2D layers (`shape[0] == shape[1]`).
    - Reference [Saxe et al.,](http://arxiv.org/abs/1312.6120)
    '''
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return B.variable(scale * q[:shape[0], :shape[1]], name=name)


def identity(shape, scale=1, name=None):
    '''Use with square 2D layers (`shape[0] == shape[1]`).
    '''

    if len(shape) != 2 or shape[0] != shape[1]:
        raise Exception('Identity matrix initialization can only be used '
                        'for 2D square matrices.')
    else:
        return B.variable(scale * np.identity(shape[0]), name=name)


def zero(shape, name=None):
    '''All zeros initialization.
    '''
    return B.zeros(shape, name=name)


def one(shape, name=None):
    '''All ones initialization.
    '''
    return B.ones(shape, name=name)

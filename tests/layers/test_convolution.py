import numpy as np
import pytest
import tensorflow as tf

from keraflow.layers import convolution
from keraflow.utils.exceptions import KeraFlowError as KError
from keraflow.utils.test_utils import layer_test


def tf_conv2d(input_val, kernel_val, strides, padding):
    x = tf.Variable(np.asarray([input_val], dtype='float32'), dtype='float32')
    kernel = tf.Variable(kernel_val, dtype='float32')
    strides = (1,) + strides + (1,)

    t_x = tf.transpose(x, [0, 2, 3, 1])
    t_kernel = tf.transpose(kernel, [2, 3, 0, 1])
    output = tf.nn.conv2d(t_x, t_kernel, strides, padding=padding.upper())
    t_output = tf.transpose(output, (0, 3, 1, 2))

    with tf.Session() as sess:
        sess.run(x.initializer)
        sess.run(kernel.initializer)
        res = sess.run(t_output)

    return res[0]


def tf_conv1d(input_val, kernel_val, stride, padding):
    input_val = input_val[None, :,:]
    strides = (stride, 1)
    p = convolution.calc_pad(padding,
                             input_val.shape[1],
                             kernel_val.shape[2],
                             strides[0])[0]
    if isinstance(p, tuple):
        p1, p2 = p
    else:
        p1 = p2 = p

    input_val = np.pad(input_val, ((0,0),(p1,p2),(0,0)), 'constant')
    res = tf_conv2d(input_val, kernel_val, strides, 'valid')
    return res[0,:,:]


# Note that we test convolution layers with exp_output given by tensorflow implementation
# Making sure that no matter which backend is used, the result will be the same.
def test_conv1D():
    settings = [(6,3,1), (6,3,2), (6,3,3)]

    def get_data(row, kernel_row, stride, padding):
        x = 1+np.arange(row)[:,None]*np.array([1]*kernel_row)
        W = np.ones((kernel_row, kernel_row))[None,None,:,:]
        b = np.array([0])
        exp_output = tf_conv1d(x, W, stride, padding)
        return x, W, b, exp_output

    # testing: padding=valid, bias=True
    # pad_num is always zero, test jsut 1 setting
    for row, kernel_row, stride in settings[:1]:
        x, W, b, exp_output = get_data(row, kernel_row, stride, 'valid')
        layer_test(convolution.Convolution1D(1, kernel_row, stride=stride, bias=True, pad='valid', initial_weights=[W, b]),
                   [x],
                   [exp_output])

    # testing: padding=same, bias=False
    # Note that we do not test serializetion here to save io.
    for row, kernel_row, stride in settings:
        x, W, b, exp_output = get_data(row, kernel_row, stride, 'same')
        layer_test(convolution.Convolution1D(1, kernel_row, stride=stride, bias=False, pad='same', initial_weights=[W]),
                   [x],
                   [exp_output],
                   test_serialization=False)


def test_conv2D():
    settings = [(6,3,(1,1)), (6,3,(1,2)), (6,3,(1,3))]

    def get_data(row, kernel_row, strides, padding):
        x = 1+np.arange(row)[:,None]*np.array([1]*row)
        x = x[None, :,:]
        W = np.ones((kernel_row, kernel_row))[None,None,:,:]
        b = np.array([0])
        exp_output = tf_conv2d(x, W, strides, padding)
        return x, W, b, exp_output

    # testing: padding=valid, bias=True
    # pad_num is always zero, test jsut 1 setting
    for row, kernel_row, strides in settings[:1]:
        x, W, b, exp_output = get_data(row, kernel_row, strides, 'valid')
        layer_test(convolution.Convolution2D(1, kernel_row, kernel_row, strides=strides, bias=True, pad='valid', initial_weights=[W, b]),
                   [x],
                   [exp_output])

    # testing: padding=same, bias=False
    # Note that we do not test serializetion here to save io.
    for row, kernel_row, strides in settings:
        x, W, b, exp_output = get_data(row, kernel_row, strides, 'same')
        layer_test(convolution.Convolution2D(1, kernel_row, kernel_row, strides=strides, bias=False, pad='same', initial_weights=[W]),
                   [x],
                   [exp_output],
                   test_serialization=False)


def test_padding1D():
    x = np.random.rand(3,3)
    pad = (1,2)
    np_pad = ((1,2),(0,0))
    layer_test(convolution.ZeroPadding1D(pad),
               [x],
               [np.pad(x, np_pad, 'constant')])


def test_padding2D():
    x = np.random.rand(3,3,3)
    pad = ((1,2),1)
    np_pad = ((0,0), (1,2),(1,1))

    layer_test(convolution.ZeroPadding2D(pad),
               [x],
               [np.pad(x, np_pad, 'constant')])

    # size could take no more than two values
    with pytest.raises(KError):
        layer_test(convolution.ZeroPadding2D(((1,2,3),1)),
                   [x],
                   [np.pad(x, np_pad, 'constant')])


# Note that we for not test exp_output for pooling layers
# because theano and tensorflow's implementation results in different baehaviors.
def test_pooling1D():
    settings = [(8,4,1), (8,4,2), (8,4,3), (8,4,4)]

    # testing: padding=valid
    # pad_num is always zero, test jsut 1 setting
    for row, pool_length, stride in settings[:1]:
        x = 1+np.arange(row)[:,None]*np.array([1]*pool_length)
        layer_test(convolution.Pooling1D('max', pool_length, pad='valid', stride=stride), [x])

        layer_test(convolution.Pooling1D('avg', pool_length, pad='valid', stride=stride), [x], debug=True)

    # testing: padding=same
    # Note that we do not test serializetion here to save io.
    for row, pool_length, stride in settings:
        x = 1+np.arange(row)[:,None]*np.array([1]*pool_length)
        layer_test(convolution.Pooling1D('max', pool_length, pad='same', stride=stride), [x], test_serialization=False)

        layer_test(convolution.Pooling1D('avg', pool_length, pad='same', stride=stride), [x], test_serialization=False)


def test_pooling2D():
    settings = [(8,4,(4,3)), (8,4,(4,2)), (8,4,(4,3)), (8,4,(4,4))]

    # testing: padding=valid
    # pad_num is always zero, test jsut 1 setting
    for row, pool_row, strides in settings[:1]:
        pool_size = (pool_row, pool_row)
        x = 1+np.arange(row)[:,None]*np.array([1]*row)
        x = x[None, :,:]

        layer_test(convolution.Pooling2D('max', pool_size, pad='valid', strides=strides),
                   [x])

        layer_test(convolution.Pooling2D('avg', pool_size, pad='valid', strides=strides),
                   [x])

    # testing: padding=same
    # Note that we do not test serializetion here to save io.
    for row, pool_row, strides in settings:
        pool_size = (pool_row, pool_row)
        x = 1+np.arange(row)[:,None]*np.array([1]*row)
        x = x[None, :,:]

        layer_test(convolution.Pooling2D('max', pool_size, pad='same', strides=strides), [x], test_serialization=False)

        layer_test(convolution.Pooling2D('avg', pool_size, pad='same', strides=strides), [x], test_serialization=False)


def test_pooling3D():
    settings = [(8,4,(4,4,1)), (8,4,(4,3,2)), (8,4,(4,2,3)), (8,4,(4,1,4)), (8,4,(4,4,4))]

    # testing: padding=valid
    # pad_num is always zero, test jsut 1 setting
    for row, pool_row, strides in settings[:1]:
        pool_size = (pool_row, pool_row, pool_row)
        x = 1+np.arange(row)[:,None]*np.array([1]*row)
        x = np.stack([int(c)*x for c in np.linspace(-100,100,row)])
        x = x[None, :,:, :]

        layer_test(convolution.Pooling3D('max', pool_size, pad='valid', strides=strides),
                   [x])

        layer_test(convolution.Pooling3D('avg', pool_size, pad='valid', strides=strides),
                   [x])

    # testing: padding=same
    # Note that we do not test serializetion here to save io.
    for row, pool_row, strides in settings:
        pool_size = (pool_row, pool_row, pool_row)
        x = 1+np.arange(row)[:,None]*np.array([1]*row)
        x = np.stack([int(c)*x for c in np.linspace(-100,100,row)])
        x = x[None, :,:, :]

        layer_test(convolution.Pooling3D('max', pool_size, pad='same', strides=strides), [x], test_serialization=False)

        layer_test(convolution.Pooling3D('avg', pool_size, pad='same', strides=strides), [x], test_serialization=False)


def test_padding3D():
    x = np.random.rand(3,3,3,3)
    pad = ((1,2), 1, 2)
    np_pad = ((0,0), (1,2), (1,1), (2,2))

    layer_test(convolution.ZeroPadding3D(pad),
               [x],
               [np.pad(x, np_pad, 'constant')])


def test_unsampling1D():
    x = np.random.rand(3,3)
    length = 2
    layer_test(convolution.UnSampling1D(length),
               [x],
               [np.repeat(x, length, axis=0)])


def test_unsampling2D():
    x = np.random.rand(3,3,3)
    size=(2,2)

    exp_output = x
    for axis in [1,2]:
        exp_output = np.repeat(exp_output, 2, axis=axis)

    layer_test(convolution.UnSampling2D(size),
               [x],
               [exp_output])


def test_unsampling3D():
    x = np.random.rand(3,3,3,3)
    size=(2,2,2)

    exp_output = x
    for axis in [1,2,3]:
        exp_output = np.repeat(exp_output, 2, axis=axis)

    layer_test(convolution.UnSampling3D(size),
               [x],
               [exp_output])


if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

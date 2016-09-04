import numpy as np
import pytest

from keraflow.layers import core
from keraflow.utils.exceptions import KeraFlowError as KError
from keraflow.utils.test_utils import layer_test

origin = np.array([[1,2,3], [4,5,6]])


def test_expand_dims():
    axis=2
    layer_test(core.ExpandDims(axis=axis),
               [origin],
               [np.expand_dims(origin, axis)])

    layer_test(core.ExpandDims(axis=axis, include_batch_dim=True),
               [origin],
               [np.expand_dims(origin, axis-1)])


def test_permute_dims():
    dims = [1, 0]
    layer_test(core.PermuteDims(dims),
               [origin],
               [np.transpose(origin, dims)])

    layer_test(core.PermuteDims([0, 2, 1], include_batch_dim=True),
               [origin],
               [np.transpose(origin, dims)])


def test_reshape():
    layer_test(core.Reshape([3, 2]),
               [origin],
               [origin.reshape([3,2])])

    layer_test(core.Reshape([3, -1]),
               [origin],
               [origin.reshape([3,-1])])

    layer_test(core.Reshape([3, 2, -1], include_batch_dim=True),
               [origin],
               origin.reshape([3, 2, -1]),
               input_args=dict(batch_size=1))

    layer_test(core.Reshape([-1, 2, 1], include_batch_dim=True),
               [origin],
               origin.reshape([3, 2, -1]),
               input_args=dict(shape=(2,3)))

    with pytest.raises(KError):
        # Trying to change batch size while but does not specify original batch size.
        layer_test(core.Reshape([3, 2, -1], include_batch_dim=True),
                   [origin],
                   origin.reshape([3, 2, -1]))

    with pytest.raises(KError):
        # more than one unknown dimension
        layer_test(core.Reshape([-1, -1]),
                   [origin],
                   [origin])

    with pytest.raises(KError):
        # dimension should >0 or =-1
        layer_test(core.Reshape([3, -2]),
                   [origin],
                   [origin])

    with pytest.raises(KError):
        # shape mismatch
        layer_test(core.Reshape([3, 3]),
                   [origin],
                   [origin])


def test_flatten():
    layer_test(core.Flatten(),
               [origin],
               [origin.flatten()])

    layer_test(core.Flatten(include_batch_dim=True),
               [origin],
               origin.flatten(),
               input_args=dict(batch_size=1))

    with pytest.raises(KError):
        # batch size not specified
        layer_test(core.Flatten(include_batch_dim=True),
                   [origin],
                   origin.flatten())

    with pytest.raises(KError):
        # input shape undetermined
        shape = list(origin.shape)
        shape[0] = None
        layer_test(core.Flatten(),
                   [origin],
                   [origin.flatten()],
                   input_args=dict(shape=tuple(shape)))


def test_repeat():
    axis = 1
    layer_test(core.Repeat(2, axis),
               [origin],
               [np.repeat(origin, 2, axis)])

    layer_test(core.Repeat(2, axis, include_batch_dim=True),
               [origin],
               [np.repeat(origin, 2, axis-1)])


def test_concatenate():
    reduced = origin[:, :-1]
    layer_test(core.Concatenate(axis=1),
               [[origin], [reduced]],
               [np.concatenate([origin, reduced], axis=1)],
               multi_input=True)

    layer_test(core.Concatenate(axis=2, include_batch_dim=True),
               [[origin], [reduced]],
               [np.concatenate([origin, reduced], axis=1)],
               multi_input=True)

    with pytest.raises(KError):
        # cant not caoncatenate tensor of different dimension
        layer_test(core.Concatenate(axis=2, include_batch_dim=True),
                   [[origin], [np.expand_dims(reduced,1)]],
                   [np.concatenate([origin, reduced], axis=1)],
                   multi_input=True)

    with pytest.raises(KError):
        # cant not caoncatenate tensor of of mismatched shape
        layer_test(core.Concatenate(axis=2, include_batch_dim=True),
                   [[origin], [origin[:-1, :-1]]],
                   [np.concatenate([origin, reduced], axis=1)],
                   multi_input=True)


def test_lambda():
    # TODO serialization has problem
    import keraflow.backend as B
    layer_test(core.Lambda(lambda x: x**2),
               [origin],
               [origin**2],
               test_serialization=False)

    layer_test(core.Lambda(lambda x: B.concatenate([x**2, x], axis=1), lambda s: (s[0],2*s[1])+s[2:]),
               [origin],
               [np.concatenate([origin**2, origin], axis=0)],
               test_serialization=False)

    shape = list(origin.shape)
    shape[0] *=2
    layer_test(core.Lambda(lambda x: B.concatenate([x**2, x], axis=1), shape),
               [origin],
               [np.concatenate([origin**2, origin], axis=0)],
               test_serialization=False)

    # output_shape_fn should be a list, a tuple, or a function
    with pytest.raises(KError):
        layer_test(core.Lambda(lambda x: x**2, 1),
                   [origin],
                   [np.concatenate([origin**2, origin], axis=0)],
                   test_serialization=False)

    # output_shape_fn should return a list or a tuple
    with pytest.raises(KError):
        layer_test(core.Lambda(lambda x: x**2, lambda x: 1),
                   [origin],
                   [np.concatenate([origin**2, origin], axis=0)],
                   test_serialization=False)


def test_activation():
    layer_test(core.Activation('tanh'),
               [origin],
               [np.tanh(origin)])


def test_element_wise_sum():
    layer_test(core.ElementWiseSum(),
               [[origin], [origin]],
               [2*origin],
               multi_input=True)


def test_element_wise_mult():
    layer_test(core.ElementWiseMult(),
               [[origin], [origin]],
               [origin**2],
               multi_input=True)


def test_dense():
    W = np.array([[1, 2]])
    b = np.array([3, 4])
    layer_test(core.Dense(2, initial_weights=[W, b]),
               [[1]],
               W+b)

    layer_test(core.Dense(2, initial_weights=[W], bias=False),
               [[1]],
               W)


def test_dropout():
    layer_test(core.Dropout(drop_rate=0.5),
               [np.ones((25, 100, 100))],
               random_exp={'mean':1,
                           'max':2,
                           'min':0})

    layer_test(core.Dropout(drop_rate=0.5),
               [np.ones((25, 100, 100))],
               [np.ones((25, 100, 100))],
               train_mode=False)

    # drop_rate must be in interval [0, 1).
    with pytest.raises(KError):
        layer_test(core.Dropout(drop_rate=-.1),
                   [np.ones((100,100, 25))],
                   random_exp={'mean':1,
                               'max':2,
                               'min':0})


def test_highway():
    W = np.array([[1, 2], [3, 4]])
    b = np.array([3, 4])
    W_carry = np.array([[5, 6], [7, 8]])
    b_carry = np.array([3, 4])
    np_input = [1, 1]

    def np_highway(bias):
        y = np.dot(np_input, W_carry)
        if bias:
            y += b_carry
        transform_weight = 1/(1+np.exp(-y))
        output = np.dot(np_input, W)
        if bias:
            output += b
        return output*transform_weight+(1-transform_weight)*np_input

    layer_test(core.Highway(initial_weights=[W, W_carry, b, b_carry]),
               [np_input],
               [np_highway(True)])

    layer_test(core.Highway(initial_weights=[W, W_carry], bias=False),
               [np_input],
               [np_highway(False)])


if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

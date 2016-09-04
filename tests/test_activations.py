import numpy as np
import pytest
from numpy.testing import assert_allclose

from keraflow import backend as B
from keraflow import activations
from keraflow.utils import KeraFlowError as KError

np_input = np.linspace(-1, 1, 5)[None,:]


def activation_test(k_fn, np_fn, np_input=np_input, debug=False):
    ndim = len(np_input.shape)
    x = B.placeholder(ndim=ndim)
    f = B.function([x], [k_fn(x)])

    if debug:
        print("Expected Output Shape:\n{}".format(np_fn(np_input)))
        print("Output:\n{}".format(f([np_input])[0]))

    assert_allclose(f([np_input])[0], np_fn(np_input), rtol=1e-05)


def test_linear():
    activation_test(lambda x: activations.linear(x)+0, lambda x: x)


def test_sigmoid():
    def sigmoid(x):
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            z = np.exp(x)
            return z / (1 + z)

    activation_test(activations.sigmoid, np.vectorize(sigmoid))


def test_hard_sigmoid():
    def hard_sigmoid(x):
        x = (x * 0.2) + 0.5
        z = 0.0 if x <= 0 else (1.0 if x >= 1 else x)
        return z

    activation_test(activations.hard_sigmoid, np.vectorize(hard_sigmoid))


def test_tanh():
    activation_test(activations.tanh, lambda x: np.tanh(x))


def test_relu():
    def relu(x):
        y = x.copy()
        y[y<0] = 0
        return y

    activation_test(activations.relu, relu)


def test_softmax():
    def softmax(values):
        e = np.max(values, axis=-1, keepdims=True)
        s = np.exp(values - e)
        return s / np.sum(s)

    activation_test(activations.softmax, softmax)

    # Testing cases of 3D input
    activation_test(activations.softmax, softmax, np_input=np.linspace(-1, 1, 5)[None,None,:])

    with pytest.raises(KError):
        # softmax accepts only 2D or 3D
        x = B.placeholder(ndim=1)
        B.function([x], [activations.softmax(x)])


def test_softplus():
    def softplus(x):
        return np.log(np.ones_like(x) + np.exp(x))

    activation_test(activations.softplus, softplus)


def test_softsign():
    def softsign(x):
        return np.divide(x, np.ones_like(x) + np.absolute(x))

    activation_test(activations.softsign, softsign)


if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

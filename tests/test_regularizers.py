import numpy as np
from keraflow import backend as B
from numpy.testing import assert_allclose
from keraflow import regularizers

np_input = np.linspace(-1, 1, 5)[None,:]
l = 0.02


def regularizer_test(reg, np_fn, np_input=np_input, debug=False):
    ndim = len(np_input.shape)
    x = B.placeholder(ndim=ndim)
    f = B.function([x], [reg(x)])

    if debug:
        print("Expected Output Shape:\n{}".format(np_fn(np_input)))
        print("Output:\n{}".format(f([np_input])[0]))

    assert_allclose(f([np_input])[0], np_fn(np_input), rtol=1e-05)


def test_l1():
    def np_L1(param):
        return np.sum(np.abs(param)) * l

    regularizer_test(regularizers.L1(l), np_L1, np_input)


def test_l2():
    def np_L2(param):
        return np.sum(np.square(param)) * l

    regularizer_test(regularizers.L2(l), np_L2, np_input)


def test_l1l2():
    def np_L1L2(param):
        return np.sum(np.abs(param)) * l + np.sum(np.square(param)) * l

    regularizer_test(regularizers.L1L2(l, l), np_L1L2, np_input)


if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

import numpy as np
import pytest

from keraflow import backend as B
from keraflow import initializations

# 2D tensor test fixture
FC_SHAPE = (100, 100)

# 4D convolution in th order. This shape has the same effective shape as FC_SHAPE
CONV_SHAPE = (25, 25, 2, 2)

# The equivalent shape of both test fixtures
SHAPE = (100, 100)


def init_test(init_fn, shapes=[FC_SHAPE, CONV_SHAPE],
              target_mean=None, target_std=None, target_max=None, target_min=None):
    for shape in shapes:
        output = B.eval(init_fn(shape))
        lim = 1e-2
        if target_std is not None:
            assert abs(output.std() - target_std) < lim
        if target_mean is not None:
            assert abs(output.mean() - target_mean) < lim
        if target_max is not None:
            assert abs(output.max() - target_max) < lim
        if target_min is not None:
            assert abs(output.min() - target_min) < lim


def test_uniform():
    init_test(initializations.uniform, target_mean=0., target_max=0.05, target_min=-0.05)


def test_normal():
    init_test(initializations.normal, target_mean=0., target_std=0.05)


def test_lecun_uniform():
    scale = np.sqrt(3. / SHAPE[0])
    init_test(initializations.lecun_uniform, target_mean=0., target_max=scale, target_min=-scale)


def test_glorot_uniform():
    scale = np.sqrt(6. / (SHAPE[0] + SHAPE[1]))
    init_test(initializations.glorot_uniform, target_mean=0., target_max=scale, target_min=-scale)


def test_glorot_normal():
    scale = np.sqrt(2. / (SHAPE[0] + SHAPE[1]))
    init_test(initializations.glorot_normal, target_mean=0., target_std=scale)


def test_he_uniform():
    scale = np.sqrt(6. / SHAPE[0])
    init_test(initializations.he_uniform, target_mean=0., target_max=scale, target_min=-scale)


def test_he_normal():
    scale = np.sqrt(2. / SHAPE[0])
    init_test(initializations.he_normal, target_mean=0., target_std=scale)


def test_orthogonal():
    init_test(initializations.orthogonal, target_mean=0.)


def test_identity():
    init_test(initializations.identity, shapes=[FC_SHAPE], target_mean=1./SHAPE[0], target_max=1.)

    with pytest.raises(Exception):
        init_test(initializations.identity, shapes=[CONV_SHAPE],target_mean=1./SHAPE[0], target_max=1.)


def test_zero():
    init_test(initializations.zero, target_max=0.)


def test_one():
    init_test(initializations.one, target_mean=1., target_max=1.)


if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

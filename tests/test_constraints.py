import numpy as np
from numpy.testing import assert_allclose

from keraflow import backend as B
from keraflow import constraints

np_input = np.array([[-1, -2], [0, 0], [1, 2], [3, 4]])


def constraint_test(k_fn, np_fn, debug=False):
    assert_allclose(B.eval(k_fn(B.variable(np_input))), np_fn(np_input), rtol=1e-05)


def test_maxnorm():
    m=2
    axis=1

    def max_norm(x):
        norms = np.sqrt(np.sum(np.square(x), axis=axis, keepdims=True))
        clipped_norms = np.clip(norms, 0, m)
        return x*clipped_norms/(1e-7+norms)

    constraint_test(constraints.MaxNorm(m=m, axis=axis), max_norm)


def test_nonneg():
    def non_neg(x):
        y = x.copy()
        y[y<0] = 0
        return y
    constraint_test(constraints.NonNeg(), non_neg)


def test_unitnorm():
    axis=1

    def unit_norm(x):
        return x/(1e-7 + np.sqrt(np.sum(np.square(x), axis=axis, keepdims=True)))

    constraint_test(constraints.UnitNorm(axis=axis), unit_norm)

if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

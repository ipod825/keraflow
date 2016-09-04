import numpy as np
from numpy.testing import assert_allclose

from keraflow.layers import core, wrappers
from keraflow.layers.base import Input
from keraflow.layers.embeddings import Embedding
from keraflow.models import Sequential
from keraflow.utils.test_utils import layer_test

origin = np.array([[[1], [2]], [[3], [4]]])


def test_TimeDistributed():

    W = np.array([[1, 2]])
    b = np.array([3, 4])
    dense = core.Dense(2, initial_weights=[W, b])

    exp_output = []
    for o_slice in origin:
        exp_output.append(np.dot([o_slice], W)+b)
    exp_output = np.concatenate(exp_output, axis=0)

    layer_test(wrappers.TimeDistributed(dense),
               origin,
               exp_output)

    # test undetermined input length
    model = Sequential()
    model.add(Input(None, dtype='int32'))
    model.add(Embedding(4, 1, initial_weights=origin.reshape(-1,1)))
    model.add(wrappers.TimeDistributed(dense))
    model.compile('sgd', 'mse')
    assert_allclose(model.predict([[0,1],[2,3]]), exp_output)


if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

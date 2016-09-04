import numpy as np
import pytest

from keraflow.layers import Dense, ElementWiseSum, Input, Layer
from keraflow.models import Sequential
from keraflow.utils.exceptions import KeraFlowError as KError

W = np.array([[1, 2]])
b = np.array([3, 4])


def create_model(**kwargs):
    model = Sequential([Input(1), Dense(2, **kwargs)])
    model.compile('sgd', 'mse')
    return model


def test_set_trainable_params():
    # set_trainable_params pattern mismatch
    with pytest.raises(KError):
        layer = Layer()
        layer.set_trainable_params('W', 1, 'b')

    # preserved name
    with pytest.raises(KError):
        d = Dense(1)
        d.set_trainable_params('trainable', 1)

    with pytest.raises(KError):
        layer = Layer()
        layer.set_trainable_params(1, 'W')


def test_get_tensor_shape():
    # should contain _keraflow_shape
    with pytest.raises(KError):
        d = Dense(1)
        d.get_tensor_shape(d)


def test_check_input_shape():
    # Class inheriting Layer does not allow mutiple inputs
    with pytest.raises(KError):
        Sequential(Dense(1)([Input(1), Input(1)]))

    # Input dimension mismatch
    with pytest.raises(KError):
        input1 = Input((1,1,1))
        Dense(1)(input1)

    # Multiple inputs layer default accepts equal shape inputs
    with pytest.raises(KError):
        input1 = Input((1,1,1))
        Dense(1)(input1)


def test_wrc_exceptions():
    # Sequential should be initialized with a list of layer
    with pytest.raises(KError):
        Sequential(Dense(2))

    # Layer weight shape mismatch
    with pytest.raises(KError):
        create_model(initial_weights={'W':np.expand_dims(W, axis=1), 'b':b})

    # regularizers does not take single input
    with pytest.raises(KError):
        create_model(initial_weights=[W, b], regularizers='l1')

    # constraints does not take single input
    with pytest.raises(KError):
        create_model(initial_weights=[W, b], constraints='maxnorm')


def test_feed_exceptions():

    # Forget to feed d1
    with pytest.raises(KError):
        d1 = Dense(1)
        Dense(1)(d1)

    # Forget to feed d1
    with pytest.raises(KError):
        d1 = Dense(1)
        Dense(1)(d1)

    # First layer of sequential should be input
    with pytest.raises(KError):
        s1 = Sequential([Dense(1)])
        s1.compile('sgd', 'mse')

    # Recursive feeding
    with pytest.raises(KError):
        input1 = Input(1)
        d = Dense(1)
        d1 = d(input1)
        d(d1)

    # Recursive feeding
    with pytest.raises(KError):
        i1 = Input(1)
        i2 = Input(1)
        i3 = Input(1)
        i4 = Input(1)
        m = ElementWiseSum()
        m1 = m([i1, i2])
        m2 = m([i3, i4])
        m([m1, m2])  # m'th output feeds to m again

    # shape should be assigned as a tuple, i.e. Input((1,2))
    with pytest.raises(KError):
        input1 = Input(1, 2)

    # You should not feed an Input layer
    with pytest.raises(KError):
        input1 = Input(1)(Input(1))

if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

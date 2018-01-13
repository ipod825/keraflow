import numpy as np
from numpy.testing import assert_allclose

import keraflow.backend as B
from keraflow.layers import Dense, Input
from keraflow.models import Sequential
from keraflow.optimizers import SGD


def test_sgd():
    ''' math:
    Let W = [A, B], b = [C, D], y = [E, F]
    MSE = 1/2*[(A+C-E)^2 + (B+D-F)^2]
    dA, dB, dC, dD = (A+C-E), (B+D-F), (A+C-E), (B+D-F)
    Assume E = 2*(A+C), F = 2*(B+D)
    dA, dB, dC, dD = -(A+C), -(B+D), -(A+C), -(B+D)
    A-=lr*dA, B-=lr*dB, C-=lr*dC, D-=lr*dD
    '''
    lr = 0.01
    W = np.array([[1, 2]])
    b = np.array([3, 4])
    wpb = W+b
    model = Sequential([Input(1), Dense(2, initial_weights=[W, b])])
    optimizer = SGD(lr=lr)
    model.compile(optimizer, 'mse')
    model.fit([1], 2*wpb, nb_epoch=1)
    expectedW = W+lr*wpb
    expectedb = (b+lr*wpb).reshape((2,))
    assert_allclose(B.eval(model.layers[1].W), expectedW)
    assert_allclose(B.eval(model.layers[1].b), expectedb)


# TODO write effecient tests to test other optimizers

if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

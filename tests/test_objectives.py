import numpy as np
from numpy.testing import assert_allclose

from keraflow import backend as B
from keraflow import objectives


def objectives_test(obj_fn, np_fn, mode=None, np_pred=None, np_true=None, debug=False):
    if mode=='reg':
        np_pred = np.array([[1,0,3], [4,8,6]])
        np_true = np.array([[1,2,4], [7,5,9]])
    elif mode=='cls':
        np_pred = np.array([[0.8, 0.9, 0.2], [0.7, 1e-9, 0.6]])
        np_true = np.array([[1, 1, 1], [0, 0, 0]])

    y_pred = B.variable(np_pred)
    y_true = B.variable(np_true)
    exp_output = np_fn(np_pred, np_true)
    if debug:
        print(obj_fn.__name__)
        print("Expected: \n{}".format(exp_output))
        print("Output: \n{}".format(B.eval(obj_fn(y_pred, y_true))))

    assert_allclose(B.eval(B.mean(obj_fn(y_pred, y_true))), np.mean(exp_output))


def clip(x, lower=B.epsilon(), upper=np.inf):
    return np.clip(x, lower, upper)


def test_accuracy():
    def cat_acc(y_pred, y_true):
        return np.expand_dims(np.equal(np.argmax(y_pred, axis=-1), np.argmax(y_true, axis=-1)), -1),

    objectives_test(objectives.accuracy,
                    cat_acc,
                    np_pred=[[0,0,.9], [0,.9,0], [.9,0,0]],
                    np_true=[[0,0,1], [0,0,1], [0,0,1]])

    def bi_acc(y_pred, y_true):
        return np.equal(np.round(y_pred), y_true)

    objectives_test(objectives.accuracy,
                    bi_acc,
                    np_pred=[[0], [0.6], [0.7]],
                    np_true=[[0], [1], [1]])


def test_square_error():
    def se(y_pred, y_true):
        return np.square(y_pred-y_true)

    objectives_test(objectives.square_error, se, mode='reg')


def test_absolute_percentage_error():
    def ape(y_pred, y_true):
        return 100*np.abs((y_pred-y_true)/clip(np.abs(y_true)))

    objectives_test(objectives.absolute_percentage_error, ape, mode='reg')


def test_squared_logarithmic_error():
    def sle(y_pred, y_true):
        logx = np.log(clip(y_pred) + 1.)
        logy = np.log(clip(y_true) + 1.)
        return np.square(logx - logy)

    objectives_test(objectives.squared_logarithmic_error, sle, mode='reg')


def test_squared_hinge():
    def sh(y_pred, y_true):
        res = 1-y_pred*y_true
        res[res<0] = 0
        return np.square(res)

    objectives_test(objectives.squared_hinge, sh, mode='cls')


def hinge():
    def hinge(y_pred, y_true):
        res = 1-y_pred*y_true
        res[res<0] = 0
        return res

    objectives_test(objectives.hinge, hinge, mode='cls')


def test_cross_entropy():

    def cat_ce(y_pred, y_true):
        y_pred = clip(y_pred, upper=1.-B.epsilon())
        y_true = clip(y_true, upper=1)
        return -np.sum(y_true * np.log(y_pred), axis=1, keepdims=True)

    def binary_ce(y_pred, y_true):
        y_true = clip(y_true, B.epsilon(), 1.-B.epsilon())
        y_pred = clip(y_pred, B.epsilon(), 1.-B.epsilon())
        return -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

    objectives_test(objectives.categorical_crossentropy,
                    cat_ce,
                    np_pred=[[.1,0,.9], [.3,.3,.4], [1,0,0]],
                    np_true=[[0,0,1], [0,0,1], [0,0,1]])

    objectives_test(objectives.binary_crossentropy,
                    binary_ce,
                    np_pred=[[0], [0.9], [0.2]],
                    np_true=[[0], [1], [1]])


def test_kullback_leibler_divergence():
    def kld(y_pred, y_true):
        y_pred = clip(y_pred, upper=1)
        y_true = clip(y_true, upper=1)
        return np.sum(y_true*np.log(y_true/y_pred), axis=-1, keepdims=True)

    objectives_test(objectives.kullback_leibler_divergence, kld, mode='cls')


def test_poisson():
    def p(y_pred, y_true):
        return y_true-y_pred*np.log(y_true+B.epsilon())

    objectives_test(objectives.poisson, p, mode='cls')


def test_cosine_proximity():
    def l2_normalize(y_pred, axis):
        norm = np.sqrt(np.sum(np.square(y_pred), axis=axis, keepdims=True))
        return y_pred / norm

    def cp(y_pred, y_true):
        y_pred = l2_normalize(y_pred, axis=-1)
        y_true = l2_normalize(y_true, axis=-1)
        return -y_pred * y_true

    objectives_test(objectives.cosine_proximity, cp, mode='reg')

if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

import numpy as np

import keraflow.backend as B
from keraflow.constraints import MaxNorm
from keraflow.layers import Input
from keraflow.models import Sequential
from keraflow.regularizers import L1
from keraflow.layers import Dense
from keraflow.utils.test_utils import assert_allclose

W = np.array([[1, 2]])
b = np.array([3, 4])
wpb = W+b
output_dim = W.shape[1]


def dense(**kwargs):
    return Dense(output_dim, **kwargs)


def create_model(layer):
    model = Sequential([Input(1), layer])
    model.compile('sgd', 'mse')
    return model


def test_trainable():
    def not_trainable(model):
        history = model.fit([[1]], 3*wpb, nb_epoch=2)
        return history[0]['loss'] == history[1]['loss']

    m1 = create_model(dense(trainable=False))
    m2 = create_model(Sequential([dense()], trainable=False))
    m3 = create_model(Sequential([Sequential([dense()])], trainable=False))

    assert not_trainable(m1)
    assert not_trainable(m2)
    assert not_trainable(m3)


def test_initial_weights():
    m1 = create_model(dense(initial_weights={'W':W, 'b':b}))
    m2 = create_model(dense(initial_weights={'W':W}))
    m3 = create_model(dense(initial_weights=[W,b]))
    m4 = create_model(dense(initial_weights=[W]))
    m5 = create_model(dense(initial_weights=W))
    m6 = create_model(dense(initial_weights=W.tolist()))
    m7 = create_model(Sequential([dense(initial_weights=[W,b])]))
    m8 = create_model(Sequential([Sequential([dense(initial_weights=[W,b])])]))

    assert (m1.predict([[1]])==wpb).all()
    assert (m2.predict([[1]])==W).all()
    assert (m3.predict([[1]])==wpb).all()
    assert (m4.predict([[1]])==W).all()
    assert (m5.predict([[1]])==W).all()
    assert (m6.predict([[1]])==W).all()
    assert (m7.predict([[1]])==wpb).all()
    assert (m8.predict([[1]])==wpb).all()


def test_regularizers():
    i = 0.01
    W_l1 = i*np.sum(abs(W))
    wpb_l1 = i*np.sum(abs(wpb))

    m1 = create_model(dense(initial_weights=[W, b]))
    m2 = create_model(dense(initial_weights=[W, b], regularizers={'W':L1(i), 'b':L1(i)}))
    m3 = create_model(dense(initial_weights=[W, b], regularizers={'W':L1(i)}))
    m4 = create_model(dense(initial_weights=[W, b], regularizers=[L1(i), L1(i)]))
    m5 = create_model(dense(initial_weights=[W, b], regularizers=[L1(i)]))
    m6 = create_model(Sequential([dense(initial_weights=[W,b], regularizers=[L1(i), L1(i)])]))
    m7 = create_model(Sequential([Sequential([dense(initial_weights=[W,b], regularizers=[L1(i), L1(i)])])]))

    def eval_model(m, train_mode=True):
        # output - expected = regularizer loss
        return m.evaluate([[1]], [wpb], train_mode=train_mode)

    assert eval_model(m1)==eval_model(m2, train_mode=False)
    assert_allclose(eval_model(m2), wpb_l1)
    assert_allclose(eval_model(m3), W_l1)
    assert_allclose(eval_model(m4), wpb_l1)
    assert_allclose(eval_model(m5), W_l1)
    assert_allclose(eval_model(m6), wpb_l1)
    assert_allclose(eval_model(m7), wpb_l1)


def test_constraints():
    maxnorm = 2
    m1 = create_model(dense(initial_weights=[W, b]))
    m2 = create_model(dense(initial_weights=[W, b], constraints={'W':MaxNorm(m=maxnorm, axis=1), 'b':MaxNorm(m=maxnorm, axis=0)}))
    m3 = create_model(dense(initial_weights=[W, b], constraints={'W':MaxNorm(m=maxnorm, axis=1)}))
    m4 = create_model(dense(initial_weights=[W, b], constraints=[MaxNorm(m=maxnorm, axis=1), MaxNorm(m=maxnorm, axis=0)]))
    m5 = create_model(dense(initial_weights=[W, b], constraints=[MaxNorm(m=maxnorm, axis=1)]))
    m6 = create_model(Sequential([dense(initial_weights=[W,b], constraints=[MaxNorm(m=maxnorm, axis=1), MaxNorm(m=maxnorm, axis=0)])]))
    m7 = create_model(Sequential([Sequential([dense(initial_weights=[W,b], constraints=[MaxNorm(m=maxnorm, axis=1), MaxNorm(m=maxnorm, axis=0)])])]))

    m1.fit([[1]], 5*wpb, nb_epoch=1)
    m2.fit([[1]], 5*wpb, nb_epoch=1)
    m3.fit([[1]], 5*wpb, nb_epoch=1)
    m4.fit([[1]], 5*wpb, nb_epoch=1)
    m5.fit([[1]], 5*wpb, nb_epoch=1)
    m6.fit([[1]], 5*wpb, nb_epoch=1)
    m7.fit([[1]], 5*wpb, nb_epoch=1)

    m1w = B.eval(m1.layers[1].W)
    m1b = B.eval(m1.layers[1].b)
    m1W_norm = np.sqrt(np.sum(np.square(m1w), axis=1))
    m1b_norm = np.sqrt(np.sum(np.square(m1b), axis=0))
    constraint_w = m1w*maxnorm/m1W_norm
    constraint_b = m1b*maxnorm/m1b_norm
    assert_allclose(B.eval(m2.layers[1].W), constraint_w)
    assert_allclose(B.eval(m2.layers[1].b), constraint_b)
    assert_allclose(B.eval(m3.layers[1].W), constraint_w)
    assert_allclose(B.eval(m3.layers[1].b), m1b)
    assert_allclose(B.eval(m4.layers[1].W), constraint_w)
    assert_allclose(B.eval(m4.layers[1].b), constraint_b)
    assert_allclose(B.eval(m5.layers[1].W), constraint_w)
    assert_allclose(B.eval(m5.layers[1].b), m1b)
    assert_allclose(B.eval(m6.layers[1].embedded_layers[0].W), constraint_w)
    assert_allclose(B.eval(m6.layers[1].embedded_layers[0].b), constraint_b)
    assert_allclose(B.eval(m7.layers[1].embedded_layers[0].embedded_layers[0].W), constraint_w)
    assert_allclose(B.eval(m7.layers[1].embedded_layers[0].embedded_layers[0].b), constraint_b)


if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

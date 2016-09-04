import inspect

import numpy as np
from numpy.testing import assert_allclose

import keraflow.backend as B
from keraflow.constraints import MaxNorm
from keraflow.layers import Activation, Dense, ElementWiseSum, Input
from keraflow.models import Model, Sequential
from keraflow.regularizers import L2
from keraflow.utils import serialize, to_list

W = np.array([[1, 2]])
b = np.array([3, 4])
wpb = W+b
output_dim = W.shape[1]

W2 = np.array([[1, 2], [3, 4]])
b2 = np.array([5, 6])
wpb2 = np.dot(wpb, W2)+b2
output_dim = W.shape[1]


def dense(**kwargs):
    return Dense(output_dim, **kwargs)


def feed_test(inp_layers, oup_layers, expected_output, num_params, multi_output=False):
    inp_layers = to_list(inp_layers)
    oup_layers = to_list(oup_layers)
    model = Model(inp_layers, oup_layers)
    model.compile('sgd', ['mse']*len(oup_layers))

    pred = model.predict([np.array([[1]])]*len(inp_layers))
    if not multi_output:
        expected_output = [expected_output]

    for p, e in zip(pred, expected_output):
        assert_allclose(p, e)

    # use caller_name to avoid race condition when conducting parallel testing
    caller_name = inspect.stack()[1][3]
    arch_fname = '/tmp/arch_{}.json'.format(caller_name)
    weight_fname = '/tmp/weight_{}.hkl'.format(caller_name)
    model.compile('sgd', ['mse']*len(oup_layers))
    if len(model.trainable_params)==0:
        weight_fname=None
    model.save_to_file(arch_fname, weight_fname, overwrite=True, indent=2)
    model2 = Model.load_from_file(arch_fname, weight_fname)
    model2.compile('sgd', ['mse']*len(oup_layers))

    assert len(model.trainable_params) == len(model2.trainable_params) == num_params

    for p1, p2 in zip(model.trainable_params, model2.trainable_params):
        assert_allclose(B.eval(p1), B.eval(p1))

    for r1, r2 in zip(model.regularizers.values(), model2.regularizers.values()):
        assert str(serialize(r1))==str(serialize(r2))

    for c1, c2 in zip(model.constraints.values(), model2.constraints.values()):
        assert str(serialize(c1))==str(serialize(c2))

params = dict(initial_weights=[W, b],
              regularizers={'b':L2(l2=0.1)},
              constraints=[MaxNorm(m=10)])


def test_single_input():
    input1 = Input(1)
    output = dense(**params)(input1)
    feed_test(input1, output, wpb, 2)


def test_multi_input_output():
    i1 = Input(1)
    i2 = Input(1)
    i3 = Input(1)
    i4 = Input(1)
    o1 = ElementWiseSum()([i1, i2])
    o2 = ElementWiseSum()([i3, i4])

    feed_test([i1, i2, i3, i4], [o1, o2], [np.array([[2]]), np.array([[2]])], 0, multi_output=True)


def test_shared_layer():
    input1 = Input(1)
    input2 = Input(1)
    shared = dense(**params)
    output = ElementWiseSum()([shared(input1), shared(input2)])
    feed_test([input1, input2], output, 2*wpb, 2)


def test_sequential_layer():
    input1 = Input(1)
    output = Sequential([dense(**params), dense(initial_weights=[W2,b2])])(input1)
    feed_test(input1, output, wpb2, 4)


def test_sequential_as_input():
    seq = Sequential([Input(1), dense(**params)])
    output = dense(initial_weights=[W2,b2])(seq)
    feed_test(seq, output, wpb2, 4)


def test_sequential_multi_input():
    input1 = Input(1)
    input2 = Input(1)
    output = Sequential([ElementWiseSum(), dense(**params)])([input1, input2])
    feed_test([input1, input2], output, 2*W+b, 2)


def test_nested_sequential():
    input1 = Input(1)
    in_seq = Sequential([dense(**params), Activation('linear')])
    output = Sequential([in_seq, dense(initial_weights=[W2, b2]), Activation('linear')])(input1)
    feed_test(input1, output, wpb2, 4)


def test_shared_sequential():
    input1 = Input(1)
    input2 = Input(1)
    shared = Sequential([dense(**params), dense(initial_weights=[W2,b2])])
    output = ElementWiseSum()([shared(input1), shared(input2)])
    feed_test([input1, input2], output, 2*wpb2, 4)


def test_shared_nested_sequential():
    input1 = Input(1)
    input2 = Input(1)
    in_seq = Sequential([dense(**params), Activation('linear')])
    seq = Sequential([in_seq, dense(initial_weights=[W2, b2]), Activation('linear')])
    output = ElementWiseSum()([seq(input1), seq(input2)])
    feed_test([input1, input2], output, 2*wpb2, 4)


def test_fit_evaluate_predict_spec():
    x = np.array([[1],[1]])
    y = np.array([[1],[1]])

    input1 = Input(1, name='input1')
    input2 = Input(1, name='input2')
    output1 = Dense(1, name='output1')(input1)
    output2 = Dense(1, name='output2')(input2)

    model0 = Model(input1, output1)
    model1 = Model([input1, input2], [output1,output2])
    model2 = Model([input1, input2], [output1,output2])

    model0.compile('sgd', 'mse', metrics=['acc'])
    model1.compile('sgd', loss=['mse','mse'], metrics=['acc'])
    model2.compile('sgd', loss={'output1':'mse', 'output2':'mse'}, metrics={'output1':'acc', 'output2':'acc'})

    model0.predict([1,1])
    model0.evaluate([1,1], [1,1])
    model0.fit([1,1], [1,1], nb_epoch=1)

    model1.predict([x,x])
    model1.evaluate([x,x], [y,y])
    model1.fit([x,x], [y,y], nb_epoch=1)

    model2.predict({'input1':x,'input2':x})
    model2.evaluate({'input1':x,'input2':x}, {'output1':y,'output2':y})
    model2.fit({'input1':x,'input2':x}, {'output1':y,'output2':y}, nb_epoch=1)


# def test_fit_evaluate_sample_weight():
#     x = np.array([[1],[1]])
#     y = np.array([[1,1],[1,1]])
#
#     input1 = Input(1, name='input1')
#     output1 = Dense(2, initial_weights=[W,b], name='output1', trainable=False)(input1)
#
#     model1 = Model(input1, output1)
#     model2 = Model(input1, output1)
#
#     model1.compile('sgd', 'mse')
#     model2.compile('sgd', 'mse')
#
#     loss_origin = model1.evaluate(x, y, sample_weights=[1,1])[0]
#     loss1 = model1.evaluate(x, y, sample_weights=[1,0])[0]
#     loss2 = model1.evaluate(x, y, sample_weights=[1,.5])[0]
#     assert loss1 == loss_origin
#     assert loss2 == loss_origin*3./4
#
#     loss_origin = model2.fit(x, y, sample_weights=[1,1], nb_epoch=1)[0]['loss']
#     loss1 = model2.fit(x, y, sample_weights=[1,0], nb_epoch=1)[0]['loss']
#     loss2 = model2.fit(x, y, sample_weights=[1,.5], nb_epoch=1)[0]['loss']
#     assert loss1 == loss_origin
#     assert loss2 == loss_origin*3./4
#
#
# def test_fit_val():
#     x = np.array([[1],[1]])
#     y = np.array([[1,1],[1,1]])
#
#     input1 = Input(1, name='input1')
#     output1 = Dense(2, initial_weights=[W,b], name='output1', trainable=False)(input1)
#
#     model1 = Model(input1, output1)
#     model2 = Model(input1, output1)
#
#     model1.compile('sgd', 'mse')
#     model2.compile('sgd', 'mse')
#
#     hist1 = model1.fit(x, y, validation_split=0.5, nb_epoch=1)
#     hist2 = model2.fit(x, y, sample_weights=[1,2], validation_split=0.5, nb_epoch=1)
#
#     assert hist1[0]['loss'] == hist1[0]['val_loss']
#     assert hist2[0]['val_loss'] == 2*hist2[0]['loss']
#
#     hist1 = model1.fit(x, y, validation_data=(x,y), nb_epoch=1)
#     hist2 = model2.fit(x, y, validation_data=(x,y,[1,2]), sample_weights=[1,1], nb_epoch=1)
#
#     assert hist1[0]['loss'] == hist1[0]['val_loss']
#     assert hist2[0]['val_loss'] == (1+2.)/2*hist2[0]['loss']

if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

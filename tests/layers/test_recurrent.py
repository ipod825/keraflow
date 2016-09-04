from keraflow.layers import recurrent
from keraflow.utils.test_utils import layer_test
import numpy as np
from keraflow import backend as B
from numpy.testing import assert_allclose

origin = np.array([[1,2,3], [4,5,6]])
input_dim = origin.shape[1]
output_dim = 2


def rnn(inputs, step, output_dim, num_states, return_sequences=False, go_backwards=False):
    outputs = []
    if go_backwards:
        inputs = [sample[::-1] for sample in inputs]

    for sample in inputs:
        outputs.append(list())
        states = [np.zeros((output_dim,))]*num_states
        for xt in sample:
            yt, states = step(xt, *states)
            outputs[-1].append(yt)

    if return_sequences:
        return outputs
    else:
        return [sample[-1] for sample in outputs]


def run_test(RNNCls, step, num_states, **cls_args):
    layer_test(RNNCls(output_dim, **cls_args),
               [origin],
               rnn([origin], step, output_dim, num_states))

    # test dropout, just test if it can pass...
    layer_test(RNNCls(output_dim, dropout_W=0.2, dropout_U=0.2, **cls_args),
               [origin], test_serialization=False)

    # test return_sequences, go_backwards, unroll
    layer_test(RNNCls(output_dim,
                      return_sequences=True,
                      go_backwards=True,
                      unroll=True,
                      **cls_args),
               [origin],
               rnn([origin],
                   step,
                   output_dim,
                   num_states,
                   return_sequences=True,
                   go_backwards=True))

    # test stateful
    rnn_layer = RNNCls(output_dim, stateful=True, **cls_args)
    exp_output = rnn([origin], step, output_dim, num_states)
    layer_test(rnn_layer,
               [origin],
               exp_output,
               input_args=dict(batch_size=1))
    assert_allclose(B.eval(rnn_layer.states[0]), exp_output)
    rnn_layer.reset_states()
    assert not np.any(B.eval(rnn_layer.states[0]))


def test_simplernn():
    def simplernn(W, U):
        def call(xt, ytm1):
            h = np.dot(xt, W)
            yt = h + np.dot(ytm1, U)
            return yt, [yt]
        return call

    W = np.ones((input_dim, output_dim))
    U = np.ones((output_dim, output_dim))
    run_test(recurrent.SimpleRNN,
             simplernn(W,U),
             num_states=1,
             activation='linear', initial_weights=[W,U])

    # test mask
    # we only test mask on SimpleRNN since the implementation is not dependent on each rnn.
    from keraflow.models import Sequential
    from keraflow.layers import Input, Embedding
    vocab_size = origin.shape[0]
    emb_dim = origin.shape[1]
    if B.name()=='tensorflow':
        input_length = origin.shape[0]
    else:
        input_length = None

    model = Sequential([])
    model.add(Input(input_length, mask_value=1))
    model.add(Embedding(vocab_size, emb_dim, initial_weights=origin))
    model.add(recurrent.SimpleRNN(output_dim, initial_weights=[W, U], activation='linear'))
    model.compile('sgd', 'mse')
    exp_output = rnn([origin[:1]], simplernn(W, U), output_dim, num_states=1)
    assert_allclose(exp_output, model.predict([[0,1]]))
    if input_length is None:
        assert_allclose(exp_output, model.predict([[0]]))


def test_gru():
    def gru(W_z, W_r, W_h, U_z, U_r, U_h):
        def call(xt, ytm1):
            x_z = np.dot(xt, W_z)
            x_r = np.dot(xt, W_r)
            x_h = np.dot(xt, W_h)
            z = x_z + np.dot(ytm1, U_z)
            r = x_r + np.dot(ytm1, U_r)

            h = (x_h + np.dot(r * ytm1, U_h))
            yt = z * ytm1 + (1 - z) * h
            return yt, [yt]
        return call

    W = [np.ones((input_dim, output_dim))]*3
    U = [np.ones((output_dim, output_dim))]*3
    run_test(recurrent.GRU,
             gru(*(W+U)),
             num_states=1,
             activation='linear',
             inner_activation='linear',
             initial_weights=W+U)


def test_lstm():
    def lstm(W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o):
        def call(xt, ytm1, ctm1):
            x_i = np.dot(xt, W_i)
            x_f = np.dot(xt, W_f)
            x_c = np.dot(xt, W_c)
            x_o = np.dot(xt, W_o)

            i = (x_i + np.dot(ytm1, U_i))
            f = (x_f + np.dot(ytm1, U_f))
            c = f * ctm1 + i * (x_c + np.dot(ytm1, U_c))
            o = (x_o + np.dot(ytm1, U_o))
            yt = o * c
            return yt, [yt, c]
        return call

    W = [np.ones((input_dim, output_dim))]*4
    U = [np.ones((output_dim, output_dim))]*4
    run_test(recurrent.LSTM,
             lstm(*(W+U)),
             num_states=2,
             activation='linear',
             inner_activation='linear',
             initial_weights=W+U)


if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

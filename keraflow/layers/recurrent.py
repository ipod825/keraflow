import numpy as np

from .. import backend as B
from .. import utils
from ..utils import KeraFlowError as KError
from .base import Layer


class Recurrent(Layer):
    '''Base class for recurrent layers. Do not use this layer in your code.
    '''
    def __init__(self, num_states, output_dim, return_sequences=False, go_backwards=False, stateful=False, unroll=False, **kwargs):
        '''
        @param num_states: int. Number of state of the layer.
        @param output_dim: int. The output dimension of the layer.
        @param return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
        @param go_backwards: Boolean. If True, process the input sequence backwards.
        @param stateful: Boolean. See below.
        @param unroll: Boolean. If True, the network will be unrolled, else a symbolic loop will be used. When using TensorFlow, the network is always unrolled, so this argument has no effect.  Unrolling can speed-up a RNN, although it tends to be more memory-intensive.  Unrolling is only suitable for short sequences.
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).

        @note
        1. You can set RNN layers to be __stateful__ by setting `stateful=True`, which means that the states computed for the samples in one batch will be reused as initial states for the samples in the next batch. This assumes a fixed batch size (specified by [Input](@ref keraflow.layers.base.Input.__init__)) and a one-to-one mapping between samples in different successive batches.
        2. All recurrent layer supports masking for input data with a variable number of timesteps. To introduce masks to your data, specify `mask_value` of [Input](@ref keraflow.layers.base.Input.__init__). Usually you will concatenate an Input layer, an embedding layer, and then a recurrent layer.
        '''
        super(Recurrent, self).__init__(**kwargs)
        self.num_states = num_states
        self.output_dim = output_dim
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.states = [None]*num_states

    def check_input_shape(self, input_shape):
        super(Recurrent, self).check_input_shape(input_shape)
        input_shape = input_shape[0]
        batch_size, timesteps = input_shape[0], input_shape[1]
        if self.stateful and batch_size is None:
            raise KError('To make recurrent layer stateful. The batch size needs to be fixed.')

        if B.name() == 'tensorflow' or (B.name() == 'theano' and self.unroll):
            if timesteps is None:
                raise KError('When using TensorFlow or theano with unroll enabled, input sequence length shold be determined. Given shape of {}: {}'.format(self.name, input_shape))

    def output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.output_dim)
        else:
            return (input_shape[0], self.output_dim)

    def input_dimension(self):
        return 3

    def _get_initial_states(self, x):
        input_shape = self.get_tensor_shape(x)
        self.batch_size = input_shape[0]
        if self.batch_size is not None:
            states = [B.zeros((self.batch_size, self.output_dim)) for _ in range(self.num_states)]
        else:
            initial_state = B.zeros_like(x[:,0,0])  # (samples,)
            initial_state = B.expand_dims(initial_state)  # (samples, 1)
            initial_state = B.tile(initial_state, [1, self.output_dim])  # (samples, output_dim)
            states = [initial_state for _ in range(self.num_states)]
        return states

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        for i in range(self.num_states):
            B.set_value(self.states[i], np.zeros((self.batch_size, self.output_dim)))

    def _step(self, x, states):
        raise NotImplementedError

    def _set_dropout(self, x):
        def gen_dropout_tensors(drop_rate, output_dim):
            num_gates = int(len(self.trainable_params)/2)
            if 0 < drop_rate < 1:
                ones = B.ones_like(B.reshape(x[:, 0, 0], (-1, 1)))  # (samples,)
                ones = B.concatenate([ones] * output_dim, 1)  # (samples, output_dim)
                res = [B.dropout(ones, drop_rate) for _ in range(num_gates)]
            else:
                res = [B.cast_to_floatx(1.) for _ in range(num_gates)]
            return utils.unlist_if_one(res)

        input_dim = self.get_tensor_shape(x)[2]
        self.D_W = gen_dropout_tensors(self.dropout_W, input_dim)
        self.D_U = gen_dropout_tensors(self.dropout_U, self.output_dim)

    def output(self, x):
        input_shape = self.get_tensor_shape(x)
        mask = self.get_tensor_mask(x)

        if self.stateful:
            self.states = initial_states = self._get_initial_states(x)
        else:
            initial_states = self._get_initial_states(x)

        self._set_dropout(x)

        last_output, outputs, states = B.rnn(self._step,
                                             inputs=x,
                                             initial_states=initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])

        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output


class SimpleRNN(Recurrent):
    '''Fully-connected RNN where the output is to be fed back to input.
    - input_shape: 3D, `(nb_samples, sequence_length, input_dim)`
    - output_shape (`return_sequences=True`): 3D, `(nb_samples, sequence_length, output_dim)`
    - output_shape (`return_sequences=False`): 2D, `(nb_samples, output_dim)`
    - parameters:
        - W: `(input_dim, output_dim)`
        - U: `(output_dim, output_dim)`
        - b: `(output_dim,)`
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform',
                 inner_init='orthogonal',
                 activation='tanh',
                 dropout_W=0.,
                 dropout_U=0.,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        '''
        @param output_dim: int. The output dimension of the layer.
        @param init: str/function. Function to initialize `W` (input to hidden transformation). See @ref Initializations.md.
        @param inner_init: str/function. Function to initialize `U` (hidden to hidden transformation). See @ref Initializations.md.
        @param activation: str/function. Activation function applied on the output. See @ref Activations.md.
        @param dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        @param dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
        @param return_sequences: Boolean. See Recurrent.__init__
        @param go_backwards: Boolean. See Recurrent.__init__
        @param stateful: Boolean. See Recurrent.__init__
        @param unroll: Boolean. See Recurrent.__init__
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).

        - Dropout reference: [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        '''

        super(SimpleRNN, self).__init__(num_states=1,
                                        output_dim=output_dim,
                                        return_sequences=return_sequences,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        unroll=unroll,
                                        **kwargs)

        self.init = utils.get_from_module('initializations', init)
        self.inner_init = utils.get_from_module('initializations', inner_init)
        self.activation = utils.get_from_module('activations', activation)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

    def init_param(self, input_shape):
        input_dim = input_shape[2]
        W = self.init((input_dim, self.output_dim), name='{}_W'.format(self.name))
        U = self.inner_init((self.output_dim, self.output_dim), name='{}_U'.format(self.name))
        b = B.zeros((self.output_dim,), name='{}_b'.format(self.name))

        self.set_trainable_params('W', W, 'U', U, 'b', b)

    def _step(self, x, states):
        prev_output = states[0]
        h = B.dot(x * self.D_W, self.W) + self.b

        output = self.activation(h + B.dot(prev_output * self.D_U, self.U))
        return output, [output]


class GRU(Recurrent):
    '''Gated Recurrent Unit - Cho et al. 2014.
    - input_shape: 3D, `(nb_samples, sequence_length, input_dim)`
    - output_shape (`return_sequences=True`): 3D, `(nb_samples, sequence_length, output_dim)`
    - output_shape (`return_sequences=False`): 2D, `(nb_samples, output_dim)`
    - parameters:
        - W_z, W_r, W_h: `(input_dim, output_dim)`
        - U_z, U_r, U_h: `(output_dim, output_dim)`
        - b_z, b_r, b_h: `(output_dim,)`
    - References:
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](http://www.aclweb.org/anthology/W14-4012)
        - [Emperical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)
    '''
    def __init__(self,
                 output_dim,
                 init='glorot_uniform',
                 inner_init='orthogonal',
                 activation='tanh',
                 inner_activation='hard_sigmoid',
                 dropout_W=0.,
                 dropout_U=0.,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        '''
        @param output_dim: int. The output dimension of the layer.
        @param init: str/function. Function to initialize `W_z`,`W_r`,`W_h`, (input to hidden transformation). See @ref Initializations.md.
        @param inner_init: str/function. Function to initialize `U_z`, `U_r`, `U_h` (hidden to hidden transformation). See @ref Initializations.md.
        @param activation: str/function. Activation function applied on the output. See @ref Activations.md.
        @param inner_activation: str/function. Activation function applied on the inner cells. See @ref Activations.md.
        @param dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        @param dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
        @param return_sequences: Boolean. See Recurrent.__init__
        @param go_backwards: Boolean. See Recurrent.__init__
        @param stateful: Boolean. See Recurrent.__init__
        @param unroll: Boolean. See Recurrent.__init__
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).

        - Dropout reference: [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        '''

        super(GRU, self).__init__(num_states=1,
                                  output_dim=output_dim,
                                  return_sequences=return_sequences,
                                  go_backwards=go_backwards,
                                  stateful=stateful,
                                  unroll=unroll,
                                  **kwargs)
        self.init = utils.get_from_module('initializations', init)
        self.inner_init = utils.get_from_module('initializations', inner_init)
        self.activation = utils.get_from_module('activations', activation)
        self.inner_activation = utils.get_from_module('activations', inner_activation)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

    def init_param(self, input_shape):
        self.input_dim = input_shape[2]

        W, U, b = {}, {}, {}
        Ws, Us, bs = [], [], []

        gates = ['z', 'r', 'h']
        for gate in gates:
            W[gate] = self.init((self.input_dim, self.output_dim), name='{}_W_{}'.format(self.name, gate))
            b[gate] = B.zeros((self.output_dim,), name='{}_b_{}'.format(self.name, gate))
            Ws += ['W_'+gate, W[gate]]
            bs += ['b_'+gate, b[gate]]

        for gate in gates:
            U[gate] = self.inner_init((self.output_dim, self.output_dim), name='{}_U_z'.format(self.name))
            Us += ['U_'+gate, U[gate]]

        self.set_trainable_params(*(Ws+Us+bs))

    def _step(self, x, states):
        h_tm1 = states[0]

        x_z = B.dot(x * self.D_W[0], self.W_z) + self.b_z
        x_r = B.dot(x * self.D_W[1], self.W_r) + self.b_r
        x_h = B.dot(x * self.D_W[2], self.W_h) + self.b_h
        z = self.inner_activation(x_z + B.dot(h_tm1 * self.D_U[0], self.U_z))
        r = self.inner_activation(x_r + B.dot(h_tm1 * self.D_U[1], self.U_r))

        hh = self.activation(x_h + B.dot(r * h_tm1 * self.D_U[2], self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]


class LSTM(Recurrent):
    '''Long-Short Term Memory unit - Hochreiter 1997. For a step-by-step description of the algorithm, see [this tutorial](http://deeplearning.net/tutorial/lstm.html).
    - input_shape: 3D, `(nb_samples, sequence_length, input_dim)`
    - output_shape (`return_sequences=True`): 3D, `(nb_samples, sequence_length, output_dim)`
    - output_shape (`return_sequences=False`): 2D, `(nb_samples, output_dim)`
    - parameters:
        - W_i, W_f, W_c, W_o: `(input_dim, output_dim)`
        - U_i, U_f, U_c, U_o: `(output_dim, output_dim)`
        - b_i, b_f, b_c, b_o: `(output_dim,)`
    - References:
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
    '''
    def __init__(self,
                 output_dim,
                 init='glorot_uniform',
                 inner_init='orthogonal',
                 forget_bias_init='one',
                 activation='tanh',
                 inner_activation='hard_sigmoid',
                 dropout_W=0.,
                 dropout_U=0.,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        '''
        @param output_dim: int. The output dimension of the layer.
        @param init: str/function. Function to initialize `W_i`,`W_f`,`W_c`, `W_o` (input to hidden transformation). See @ref Initializations.md.
        @param inner_init: str/function. Function to initialize `U_i`,`U_f`,`U_c`, `U_o` (hidden to hidden transformation). See @ref Initializations.md.
        @param forget_bias_init: initialization function for the bias of the forget gate. [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf) recommend initializing with ones.
        @param activation: str/function. Activation function applied on the output. See @ref Activations.md.
        @param inner_activation: str/function. Activation function applied on the inner cells. See @ref Activations.md.
        @param dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        @param dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
        @param return_sequences: Boolean. See Recurrent.__init__
        @param go_backwards: Boolean. See Recurrent.__init__
        @param stateful: Boolean. See Recurrent.__init__
        @param unroll: Boolean. See Recurrent.__init__
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).

        - Dropout reference: [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        '''

        super(LSTM, self).__init__(num_states=2,
                                   output_dim=output_dim,
                                   return_sequences=return_sequences,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)
        self.init = utils.get_from_module('initializations', init)
        self.inner_init = utils.get_from_module('initializations', inner_init)
        self.forget_bias_init = utils.get_from_module('initializations', forget_bias_init)
        self.activation = utils.get_from_module('activations', activation)
        self.inner_activation = utils.get_from_module('activations', inner_activation)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

    def init_param(self, input_shape):
        self.input_dim = input_shape[2]

        W, U, b = {}, {}, {}
        Ws, Us, bs = [], [], []

        gates = ['i', 'f', 'c', 'o']
        for gate in gates:
            W[gate] = self.init((self.input_dim, self.output_dim), name='{}_W_{}'.format(self.name, gate))
            b[gate] = B.zeros((self.output_dim,), name='{}_b_{}'.format(self.name, gate))
            Ws += ['W_'+gate, W[gate]]
            bs += ['b_'+gate, b[gate]]

        for gate in gates:
            U[gate] = self.inner_init((self.output_dim, self.output_dim), name='{}_U_z'.format(self.name))
            Us += ['U_'+gate, U[gate]]

        self.set_trainable_params(*(Ws+Us+bs))

    def _step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]

        x_i = B.dot(x * self.D_W[0], self.W_i) + self.b_i
        x_f = B.dot(x * self.D_W[1], self.W_f) + self.b_f
        x_c = B.dot(x * self.D_W[2], self.W_c) + self.b_c
        x_o = B.dot(x * self.D_W[3], self.W_o) + self.b_o

        i = self.inner_activation(x_i + B.dot(h_tm1 * self.D_U[0], self.U_i))
        f = self.inner_activation(x_f + B.dot(h_tm1 * self.D_U[1], self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + B.dot(h_tm1 * self.D_U[2], self.U_c))
        o = self.inner_activation(x_o + B.dot(h_tm1 * self.D_U[3], self.U_o))

        h = o * self.activation(c)
        return h, [h, c]

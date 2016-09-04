import numpy as np
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import pool

from .common import _FLOATX, Backend

try:
    from theano.tensor.nnet.nnet import softsign as T_softsign
except ImportError:
    from theano.sandbox.softsign import softsign as T_softsign


# INTERNAL UTILS

# VARIABLE MANIPULATION


class TheanoBackend(Backend):

    def __init__(self, **kwargs):
        super(TheanoBackend, self).__init__(**kwargs)
        self.rng = RandomStreams(self._seed)
        theano.config.floatX = _FLOATX

    def reset_random_state(self):
        self.rng = RandomStreams(self._seed)

    # TENSOR CREATION

    def constant(self, value, dtype=_FLOATX, name=None):
        return T.constant(value, dtype=dtype, name=name)

    def variable(self, value, dtype=_FLOATX, name=None):
        '''Instantiate a tensor variable.
        '''
        value = np.asarray(value, dtype=dtype)
        return theano.shared(value=value, name=name, strict=False)

    def placeholder(self, shape=None, ndim=None, dtype=_FLOATX, name=None):
        '''Instantiate an input data placeholder variable.
        '''
        if shape is None and ndim is None:
            return T.scalar(name, dtype=dtype)
        else:
            if shape is None:
                shape = tuple([None for _ in range(ndim)])
            broadcast = (False,) * len(shape)
            return T.TensorType(dtype, broadcast)(name)

    def patternbroadcast(self, x, pattern):
        return T.patternbroadcast(x, pattern)

    # TENSOR OPERATION

    def shape(self, x):
        '''Returns the symbolic shape of a tensor.
        '''
        return x.shape

    def ndim(self, x):
        return x.ndim

    def dtype(self, x):
        return x.dtype

    def eval(self, x):
        '''Evaluates the value of a tensor.
        Returns a Numpy array.
        '''
        return np.asarray(x.eval())

    def cast(self, x, dtype):
        return T.cast(x, dtype)

    def set_value(self, x, value):
        x.set_value(np.asarray(value, dtype=x.dtype))

    def switch(self, condition, then_expression, else_expression):
        '''condition: scalar tensor.
        '''
        return T.switch(condition, then_expression, else_expression)

    def _Function(self, inputs, outputs, updates=[], **kwargs):
        function = theano.function(inputs, outputs, updates=updates,
                                   allow_input_downcast=True,
                                   on_unused_input='warn',
                                   **kwargs)

        def call(input_values):
            assert type(inputs) in {list, tuple}
            return function(*input_values)

        return call

    def function(self, inputs, outputs, updates=[], **kwargs):
        return self._Function(inputs, outputs, updates=updates, **kwargs)

    def gradients(self, loss, variables):
        return T.grad(loss, variables)

    # RANDOMNESS

    def random_normal(self, shape, mean=0.0, std=1.0, dtype=_FLOATX):
        return self.rng.normal(size=shape, avg=mean, std=std, dtype=dtype)

    def random_uniform(self, shape, low=0.0, high=1.0, dtype=_FLOATX):
        return self.rng.uniform(shape, low=low, high=high, dtype=dtype)

    def random_binomial(self, shape, p=0.0, dtype=_FLOATX):
        return self.rng.binomial(shape, p=p, dtype=dtype)

    # NUMPY API

    def zeros(self, shape, dtype=_FLOATX, name=None):
        '''Instantiate an all-zeros tensor variable.
        '''
        return self.variable(np.zeros(shape), dtype, name)

    def ones(self, shape, dtype=_FLOATX, name=None):
        '''Instantiate an all-ones tensor variable.
        '''
        return self.variable(np.ones(shape), dtype, name)

    def eye(self, size, dtype=_FLOATX, name=None):
        '''Instantiate an identity matrix.
        '''
        return self.variable(np.eye(size), dtype, name)

    def zeros_like(self, x):
        '''Instantiates an all-zeros tensor of the same shape as another tensor.
        '''
        return T.zeros_like(x)

    def ones_like(self, x):
        '''Instantiates an all-ones tensor of the same shape as another tensor.
        '''
        return T.ones_like(x)

    def dot(self, x, y):
        '''numpy.dot on tensors
        '''
        return T.dot(x, y)

    def gather(self, reference, indices):
        '''Retrieves the vectors of indices `indices` in the 2D tensor `reference`.
        @param reference: a 2D tensor.
        @param indices: 2D int tensor or list.
        @return 3D tensor.
        '''
        return reference[indices]

    def max(self, x, axis=None, keepdims=False):
        return T.max(x, axis=axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        return T.min(x, axis=axis, keepdims=keepdims)

    def sum(self, x, axis=None, keepdims=False):
        return T.sum(x, axis=axis, keepdims=keepdims)

    def prod(self, x, axis=None, keepdims=False):
        '''Multiply the values in a tensor, alongside the specified axis.
        '''
        return T.prod(x, axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        dtype = None
        if 'int' in x.dtype:
            dtype = _FLOATX
        return T.mean(x, axis=axis, keepdims=keepdims, dtype=dtype)

    def std(self, x, axis=None, keepdims=False):
        return T.std(x, axis=axis, keepdims=keepdims)

    def any(self, x, axis=None, keepdims=False):
        '''Bitwise reduction (logical OR).
        '''
        return T.any(x, axis=axis, keepdims=keepdims)

    def argmax(self, x, axis=-1):
        return T.argmax(x, axis=axis, keepdims=False)

    def argmin(self, x, axis=-1):
        return T.argmin(x, axis=axis, keepdims=False)

    def square(self, x):
        return T.sqr(x)

    def abs(self, x):
        return T.abs_(x)

    def sqrt(self, x):
        x = self.clip(x, 0., np.inf)
        return T.sqrt(x)

    def exp(self, x):
        return T.exp(x)

    def log(self, x):
        return T.log(x)

    def round(self, x):
        return T.round(x)

    def sign(self, x):
        return T.sgn(x)

    def pow(self, x, a):
        return T.pow(x, a)

    def clip(self, x, min_value, max_value):
        if max_value < min_value:
            max_value = min_value
        return T.clip(x, min_value, max_value)

    def equal(self, x, y):
        return T.eq(x, y)

    def not_equal(self, x, y):
        return T.neq(x, y)

    def maximum(self, x, y):
        return T.maximum(x, y)

    def minimum(self, x, y):
        return T.minimum(x, y)

    def sin(self, x):
        return T.sin(x)

    def cos(self, x):
        return T.cos(x)

    def concatenate(self, tensors, axis=-1):
        return T.concatenate(tensors, axis=axis)

    def reshape(self, x, shape):
        return T.reshape(x, shape)

    def transpose(self, x, dims):
        '''Transpose dimensions.

        dims should be a tuple or list of
        dimension indices, e.g. [0, 2, 1].
        '''
        dims = tuple(dims)
        return x.dimshuffle(dims)

    def repeat(self, x, rep, axis):
        '''Repeat the elements of a tensor along an axis, like np.repeat.

        If x has shape (s1, s2, s3) and axis=1, the output
        will have shape (s1, s2 * rep, s3).
        '''
        return T.repeat(x, rep, axis=axis)

    def tile(self, x, n):
        return T.tile(x, n)

    def flatten(self, x):
        return T.flatten(x)

    def expand_dims(self, x, axis=-1):
        '''Add a 1-sized dimension at index "axis".
        '''
        pattern = [i for i in range(x.type.ndim)]
        if axis < 0:
            if x.type.ndim == 0:
                axis = 0
            else:
                axis = axis % x.type.ndim + 1
        pattern.insert(axis, 'x')
        return x.dimshuffle(pattern)

    def squeeze(self, x, axis):
        '''Remove a 1-dimension from the tensor at index "axis".
        '''
        x = T.addbroadcast(x, axis)
        return T.squeeze(x)

    def stack(self, x):
        return T.stack(*x)

    # NN OPERATIONS

    def sigmoid(self, x):
        return T.nnet.sigmoid(x)

    def hard_sigmoid(self, x):
        return T.nnet.hard_sigmoid(x)

    def tanh(self, x):
        return T.tanh(x)

    def relu(self, x, alpha=0., max_value=None):
        x = T.nnet.relu(x, alpha)
        if max_value is not None:
            x = T.minimum(x, max_value)
        return x

    def softmax(self, x):
        return T.nnet.softmax(x)

    def softplus(self, x):
        return T.nnet.softplus(x)

    def softsign(self, x):
        return T_softsign(x)

    def dropout(self, x, drop_rate, noise_shape=None):
        '''Sets entries in `x` to zero at random, while scaling the entire tensor.
        @param x: tensor
        @param drop_rate: fraction of the entries in the tensor that will be set to 0.
        @param noise_shape: shape for randomly generated keep/drop flags, must be broadcastable to the shape of `x`
        '''
        assert drop_rate > 0. or drop_rate < 1, 'Dropout drop_rate must be in interval [0, 1].'

        retain_prob = 1. - drop_rate

        if noise_shape is None:
            random_tensor = self.random_binomial(self.shape(x), p=retain_prob)
        else:
            random_tensor = self.random_binomial(noise_shape, p=retain_prob)
            random_tensor = T.patternbroadcast(random_tensor, [dim == 1 for dim in noise_shape])

        train_x = x*random_tensor
        train_x /= retain_prob
        return self.in_train_phase(train_x, x)

    def conv2d(self, x, kernel, strides=(1, 1), input_shape=None, filter_shape=None):
        # kernel shape: (input_depth, output_depth, k_rows, k_cols)
        # TH kernel shape: (output_depth, input_depth, k_rows, k_cols)
        kernel = self.transpose(kernel, [1, 0, 2, 3])
        filter_shape = list(np.asarray(filter_shape)[[1, 0, 2, 3]])

        x = T.nnet.conv2d(x, kernel,
                          border_mode='valid',
                          subsample=strides,
                          input_shape=input_shape,
                          filter_shape=filter_shape)

        return x

    def pool(self, x, mode, pool_size, strides, padding=(0,0)):

        if strides is None:
            strides = pool_size
        assert len(strides)==len(pool_size)
        do2D = len(pool_size)==2

        if mode=='avg':
            mode='average_exc_pad'

        # theano requires symmetric padding
        # We pad the larger on when two sides' padding are unequal
        max_padding = list(padding)
        for i, p in enumerate(padding):
            if isinstance(p, tuple):
                assert p[1]==p[0]+1
                max_padding[i] = p[1]
            else:
                max_padding[i] = p

        if do2D:
            pool_out = pool.pool_2d(x, ds=pool_size, st=strides,
                                    ignore_border=True,
                                    padding=max_padding,
                                    mode=mode)
        else:
            # pool over HW
            pool_out = pool.pool_2d(x.dimshuffle(0,1,4,2,3),
                                    ds=pool_size[:2],
                                    st=strides[:2],
                                    ignore_border=True,
                                    padding=max_padding[:2],
                                    mode=mode)

            # pool over Z
            pool_out = pool.pool_2d(pool_out.dimshuffle(0,1,3,4,2),
                                    ds=(1,pool_size[2]),
                                    st=(1,strides[2]),
                                    ignore_border=True,
                                    padding=(0, max_padding[2]),
                                    mode=mode)

        # theano might output more than expected output shape (due to max padding). We truncate them here
        exp_l = []
        for i in range(len(strides)):
            l = T.ceil(self.cast(x.shape[i+2], _FLOATX)/strides[i])
            exp_l.append(self.cast(l, 'int32'))

        if do2D:
            return pool_out[:, :, :exp_l[0], :exp_l[1]]
        else:
            return pool_out[:, :, :exp_l[0], :exp_l[1], :exp_l[2]]

    def padding(self, x, pad, pad_dims, output_shape):
        # x shape: (nb_sample, input_depth, rows, cols)

        output = T.zeros((x.shape[0],) + output_shape[1:])
        indices = [slice(None), slice(None)]  # nb_sample, input_depth does not change

        for i in range(2, len(output_shape)):
            if i not in pad_dims:
                indices.append(slice(None))
            else:
                p = pad[i-2]
                if isinstance(p, (tuple,list)):
                    assert len(p)==2
                    assert p[0]!=0 or p[1]!=0
                    indices.append(slice(p[0], -p[1]))
                else:
                    if p==0:
                        indices.append(slice(None))
                    else:
                        indices.append(slice(p, -p))

        return T.set_subtensor(output[indices], x)

    def rnn(self, step_function, inputs, initial_states,
            go_backwards=False, mask=None,
            unroll=False, input_length=None):
        '''Iterates over the time dimension of a tensor.

        # Arguments
            inputs: tensor of temporal data of shape (samples, time, ...)
                (at least 3D).
            step_function:
                Parameters:
                    input: tensor with shape (samples, ...) (no time dimension),
                        representing input for the batch of samples at a certain
                        time step.
                    states: list of tensors.
                Returns:
                    output: tensor with shape (samples, ...) (no time dimension),
                    new_states: list of tensors, same length and shapes
                        as 'states'.
            initial_states: tensor with shape (samples, ...) (no time dimension),
                containing the initial values for the states used in
                the step function.
            go_backwards: boolean. If True, do the iteration over
                the time dimension in reverse order.
            mask: binary tensor with shape (samples, time),
                with a zero for every element that is masked.
            unroll: whether to unroll the RNN or to use a symbolic loop (`scan`).
            input_length: must be specified if using `unroll`.

        # Returns
            A tuple (last_output, outputs, new_states).
                last_output: the latest output of the rnn, of shape (samples, ...)
                outputs: tensor with shape (samples, time, ...) where each
                    entry outputs[s, t] is the output of the step function
                    at time t for sample s.
                new_states: list of tensors, latest states returned by
                    the step function, of shape (samples, ...).
        '''
        ndim = inputs.ndim
        assert ndim >= 3, 'Input should be at least 3D.'

        if unroll:
            if input_length is None:
                raise Exception('When specifying `unroll=True`, an `input_length` '
                                'must be provided to `rnn`.')

        axes = [1, 0] + list(range(2, ndim))
        inputs = inputs.dimshuffle(axes)

        if mask is not None:
            if mask.ndim == ndim-1:
                mask = self.expand_dims(mask)
            assert mask.ndim == ndim
            mask = mask.dimshuffle(axes)

            if unroll:
                indices = list(range(input_length))
                if go_backwards:
                    indices = indices[::-1]

                successive_outputs = []
                successive_states = []
                states = initial_states
                for i in indices:
                    output, new_states = step_function(inputs[i], states)

                    if len(successive_outputs) == 0:
                        prev_output = self.zeros_like(output)
                    else:
                        prev_output = successive_outputs[-1]

                    output = T.switch(mask[i], output, prev_output)
                    kept_states = []
                    for state, new_state in zip(states, new_states):
                        kept_states.append(T.switch(mask[i], new_state, state))
                    states = kept_states

                    successive_outputs.append(output)
                    successive_states.append(states)

                outputs = T.stack(*successive_outputs)
                states = []
                for i in range(len(successive_states[-1])):
                    states.append(T.stack(*[states_at_step[i] for states_at_step in successive_states]))
            else:
                # build an all-zero tensor of shape (samples, output_dim)
                initial_output = step_function(inputs[0], initial_states)[0] * 0
                # Theano gets confused by broadcasting patterns in the scan op
                initial_output = T.unbroadcast(initial_output, 0, 1)

                def _step(input, mask, output_tm1, *states):
                    output, new_states = step_function(input, states)
                    # output previous output if masked.
                    output = T.switch(mask, output, output_tm1)
                    return_states = []
                    for state, new_state in zip(states, new_states):
                        return_states.append(T.switch(mask, new_state, state))
                    return [output] + return_states

                results, _ = theano.scan(
                    _step,
                    sequences=[inputs, mask],
                    outputs_info=[initial_output] + initial_states,
                    go_backwards=go_backwards)

                # deal with Theano API inconsistency
                if type(results) is list:
                    outputs = results[0]
                    states = results[1:]
                else:
                    outputs = results
                    states = []
        else:
            if unroll:
                indices = list(range(input_length))
                if go_backwards:
                    indices = indices[::-1]

                successive_outputs = []
                successive_states = []
                states = initial_states
                for i in indices:
                    output, states = step_function(inputs[i], states)
                    successive_outputs.append(output)
                    successive_states.append(states)
                outputs = T.stack(*successive_outputs)
                states = []
                for i in range(len(successive_states[-1])):
                    states.append(T.stack(*[states_at_step[i] for states_at_step in successive_states]))

            else:
                def _step(input, *states):
                    output, new_states = step_function(input, states)
                    return [output] + new_states

                results, _ = theano.scan(
                    _step,
                    sequences=inputs,
                    outputs_info=[None] + initial_states,
                    go_backwards=go_backwards)

                # deal with Theano API inconsistency
                if type(results) is list:
                    outputs = results[0]
                    states = results[1:]
                else:
                    outputs = results
                    states = []

        outputs = T.squeeze(outputs)
        last_output = outputs[-1]

        axes = [1, 0] + list(range(2, outputs.ndim))
        outputs = outputs.dimshuffle(axes)
        states = [T.squeeze(state[-1]) for state in states]
        return last_output, outputs, states

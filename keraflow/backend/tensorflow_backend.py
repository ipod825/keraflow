import copy
import os
import warnings

import numpy as np
import tensorflow as tf

from ..utils import KeraFlowError as KError
from .common import _FLOATX, Backend


class TensorflowBackend(Backend):

    def __init__(self, **kwargs):
        super(TensorflowBackend, self).__init__(**kwargs)
        if tf.get_default_session() is not None:
            self.session = tf.get_default_session()
        else:
            if not os.environ.get('OMP_NUM_THREADS'):
                self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            else:
                nb_thread = int(os.environ.get('OMP_NUM_THREADS'))
                self.session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=nb_thread,
                                                                allow_soft_placement=True))

    def reset_random_state(self):
        tf.set_random_seed(self._seed)

    def __del__(self):
        # tf.reset_default_graph()
        # self.session = None
        pass

    # SYMBOLIC TENSOR

    def constant(self, value, dtype=_FLOATX, name=None):
        return tf.constant(value, dtype=dtype, name=name)

    def variable(self, value, dtype=_FLOATX, name=None):
        v = tf.Variable(np.asarray(value, dtype=dtype), name=name)
        self.session.run(v.initializer)
        return v

    def placeholder(self, shape=None, ndim=None, dtype=_FLOATX, name=None):
        if not shape:
            if ndim:
                shape = tuple([None for _ in range(ndim)])
        x = tf.placeholder(dtype, shape=shape, name=name)
        return x

    def patternbroadcast(self, x, pattern):
        return x

    # TENSOR OPERATION

    def shape(self, x):
        '''Returns the symbolic shape of a tensor.
        '''
        return tf.shape(x)

    def int_shape(self, x):
        '''Returns the shape of a tensor as a tuple of
        integers or None entries.
        Note that this function only works with TensorFlow.
        '''
        shape = x.get_shape()
        return tuple([i.__int__() for i in shape])

    def ndim(self, x):
        '''Returns the number of axes in a tensor, as an integer.
        '''
        dims = x.get_shape()._dims
        if dims is not None:
            return len(dims)
        return None

    def dtype(self, x):
        '''Returns the dtype of a tensor, as a string.
        '''
        return x.dtype.name

    def eval(self, x):
        '''Evaluates the value of a tensor.
        Returns a Numpy array.
        '''
        return x.eval(session=self.session)

    def cast(self, x, dtype):
        '''Casts a tensor to a different dtype.
        '''
        return tf.cast(x, dtype)

    def set_value(self, x, value):
        '''Sets the value of a tensor variable,
        from a Numpy array.
        '''
        tf.assign(x, np.asarray(value)).op.run(session=self.session)

    def switch(self, condition, then_expression, else_expression):
        '''Switches between two operations depending on a scalar value (int or bool).
        Note that both `then_expression` and `else_expression`
        should be symbolic tensors of the *same shape*.

        # Arguments
            condition: scalar tensor.
            then_expression: TensorFlow operation.
            else_expression: TensorFlow operation.
        '''
        x_shape = copy.copy(then_expression.get_shape())
        x = tf.python.control_flow_ops.cond(self.cast(condition, 'bool'),
                                            lambda: then_expression,
                                            lambda: else_expression)
        x.set_shape(x_shape)
        return x

    def _Function(self, inputs, outputs, updates=[]):
        for item, name in zip([inputs,outputs,updates], ['Inputs','Outputs','Updates']):
            if not type(item) in (list, tuple):
                raise KError('{}  to a TensorFlow backend function should be a list or tuple.'.format(name))
        inputs = list(inputs)
        outputs = list(outputs)

        with tf.control_dependencies(outputs):
            updates = [tf.assign(p, new_p) for (p, new_p) in updates]

        def call(input_values):
            assert type(input_values) in {list, tuple}
            names = [v.name for v in inputs]
            feed_dict = dict(zip(names, input_values))
            session = self.session
            updated = session.run(outputs + updates, feed_dict=feed_dict)
            return updated[:len(outputs)]

        return call

    def function(self, inputs, outputs, updates=[], **kwargs):
        '''Instantiates a Keraflow function.

        # Arguments
            inputs: list of placeholder/variable tensors.
            outputs: list of output tensors.
            updates: list of update tuples (old_tensor, new_tensor).
        '''
        if len(kwargs) > 0:
            msg = [
                "Expected no kwargs, you passed %s" % len(kwargs),
                "kwargs passed to function are ignored with Tensorflow backend"
            ]
            warnings.warn('\n'.join(msg))
        return self._Function(inputs, outputs, updates=updates)

    def gradients(self, loss, variables):
        '''Returns the gradients of `variables` (list of tensor variables)
        with regard to `loss`.
        '''
        return tf.gradients(loss, variables)

    # RANDOMNESS

    def random_normal(self, shape, mean=0.0, std=1.0, dtype=_FLOATX):
        return tf.random_normal(shape, mean=mean, stddev=std, dtype=dtype)

    def random_uniform(self, shape, low=0.0, high=1.0, dtype=_FLOATX):
        return tf.random_uniform(shape, minval=low, maxval=high, dtype=dtype)

    def random_binomial(self, shape, p=0.0, dtype=_FLOATX):
        return tf.select(tf.random_uniform(shape, dtype=dtype) <= p, tf.ones(shape), tf.zeros(shape))

    # NUMPY API

    def zeros(self, shape, dtype=_FLOATX, name=None):
        '''Instantiates an all-zeros tensor variable.
        '''
        return self.variable(np.zeros(shape), dtype, name)

    def ones(self, shape, dtype=_FLOATX, name=None):
        '''Instantiates an all-ones tensor variable.
        '''
        return self.variable(np.ones(shape), dtype, name)

    def eye(self, size, dtype=_FLOATX, name=None):
        '''Instantiate an identity matrix.
        '''
        return self.variable(np.eye(size), dtype, name)

    def zeros_like(self, x, name=None):
        '''Instantiates an all-zeros tensor of the same shape as another tensor.
        '''
        return tf.zeros_like(x, name=name)

    def ones_like(self, x, name=None):
        '''Instantiates an all-ones tensor of the same shape as another tensor.
        '''
        return tf.ones_like(x, name=name)

    def dot(self, x, y):
        '''numpy.dot on tensors
        '''
        if self.ndim(x) is not None and (self.ndim(x) > 2 or self.ndim(y) > 2):
            x_shape = (-1,) + self.int_shape(x)[1:]
            y_shape = self.int_shape(y)
            y_permute_dim = list(range(self.ndim(y)))
            y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
            xt = tf.reshape(x, [-1, x_shape[-1]])
            yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
            return tf.reshape(tf.matmul(xt, yt), x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
        out = tf.matmul(x, y)
        return out

    def gather(self, reference, indices):
        '''Retrieves the vectors of indices `indices` in the 2D tensor `reference`.
        @param reference: a 2D tensor.
        @param indices: 2D int tensor or list.
        @return 3D tensor.
        '''
        return tf.gather(reference, indices)

    def max(self, x, axis=None, keepdims=False):
        return tf.reduce_max(x, reduction_indices=axis, keep_dims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        return tf.reduce_min(x, reduction_indices=axis, keep_dims=keepdims)

    def sum(self, x, axis=None, keepdims=False):
        return tf.reduce_sum(x, reduction_indices=axis, keep_dims=keepdims)

    def prod(self, x, axis=None, keepdims=False):
        '''Multiplies the values in a tensor, alongside the specified axis.
        '''
        return tf.reduce_prod(x, reduction_indices=axis, keep_dims=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        '''Mean of a tensor, alongside the specificied axis.
        '''
        if x.dtype.base_dtype == tf.bool:
            x = self.cast(x, _FLOATX)
        return tf.reduce_mean(x, reduction_indices=axis, keep_dims=keepdims)

    def std(self, x, axis=None, keepdims=False):
        '''Standard deviation of a tensor, alongside the specificied axis.
        '''
        if x.dtype.base_dtype == tf.bool:
            x = self.cast(x, _FLOATX)
        m = tf.reduce_mean(x, reduction_indices=axis, keep_dims=True)
        devs_squared = tf.square(x - m)
        return tf.sqrt(tf.reduce_mean(devs_squared,
                                      reduction_indices=axis,
                                      keep_dims=keepdims))

    def any(self, x, axis=None, keepdims=False):
        '''Bitwise reduction (logical OR).

        Returns an uint8 tensor (0s and 1s).
        '''
        x = self.cast(x, tf.bool)
        x = tf.reduce_any(x, reduction_indices=axis, keep_dims=keepdims)
        return self.cast(x, tf.uint8)

    def argmax(self, x, axis=-1):
        '''Returns the index of the maximum value
        along a tensor axis.
        '''
        if axis < 0:
            axis = axis % len(x.get_shape())
        return tf.argmax(x, axis)

    def argmin(self, x, axis=-1):
        '''Returns the index of the minimum value
        along a tensor axis.
        '''
        if axis < 0:
            axis = axis % len(x.get_shape())
        return tf.argmin(x, axis)

    def square(self, x):
        '''Element-wise square.
        '''
        return tf.square(x)

    def abs(self, x):
        '''Element-wise absolute value.
        '''
        return tf.abs(x)

    def sqrt(self, x):
        '''Element-wise square root.
        '''
        x = self.clip(x, 0, np.inf)
        return tf.sqrt(x)

    def exp(self, x):
        '''Element-wise exponential.
        '''
        return tf.exp(x)

    def log(self, x):
        '''Element-wise log.
        '''
        return tf.log(x)

    def round(self, x):
        '''Element-wise rounding to the closest integer.
        '''
        return tf.round(x)

    def sign(self, x):
        '''Element-wise sign.
        '''
        return tf.sign(x)

    def pow(self, x, a):
        '''Element-wise exponentiation.
        '''
        return tf.pow(x, a)

    def clip(self, x, min_value, max_value):
        '''Element-wise value clipping.
        '''
        if max_value < min_value:
            max_value = min_value
        return tf.clip_by_value(x, self.cast(min_value, dtype=_FLOATX),
                                self.cast(max_value, dtype=_FLOATX))

    def equal(self, x, y):
        '''Element-wise equality between two tensors.
        Returns a bool tensor.
        '''
        return tf.equal(x, y)

    def not_equal(self, x, y):
        '''Element-wise inequality between two tensors.
        Returns a bool tensor.
        '''
        return tf.not_equal(x, y)

    def gt(self, x, y):
        return tf.greater(x, y)

    def lt(self, x, y):
        return tf.less(x, y)

    def maximum(self, x, y):
        '''Element-wise maximum of two tensors.
        '''
        return tf.maximum(x, y)

    def minimum(self, x, y):
        '''Element-wise minimum of two tensors.
        '''
        return tf.minimum(x, y)

    def sin(self, x):
        '''Computes sin of x element-wise.
        '''
        return tf.sin(x)

    def cos(self, x):
        '''Computes cos of x element-wise.
        '''
        return tf.cos(x)

    def concatenate(self, tensors, axis=-1):
        '''Concantes a list of tensors alongside the specified axis.
        '''
        if axis < 0:
            if len(tensors[0].get_shape()):
                axis = axis % len(tensors[0].get_shape())
            else:
                axis = 0
        return tf.concat(axis, tensors)

    def reshape(self, x, shape):
        '''Reshapes a tensor to the specified shape.
        '''
        return tf.reshape(x, shape)

    def transpose(self, x, dims):
        '''Permutes axes in a tensor.

        # Arguments
            dims: should be a tuple of
                dimension indices, e.g. (0, 2, 1).
        '''
        return tf.transpose(x, perm=dims)

    def repeat(self, x, rep, axis):
        '''Repeats the elements of a tensor along an axis, like np.repeat

        If x has shape (s1, s2, s3) and axis=1, the output
        will have shape (s1, s2 * rep, s3)
        '''
        x_shape = x.get_shape().as_list()
        # slices along the repeat axis
        splits = tf.split(axis, x_shape[axis], x)
        # repeat each slice the given number of reps
        x_rep = [s for s in splits for i in range(rep)]
        return tf.concat(axis, x_rep)

    def tile(self, x, n):
        if not hasattr(n, 'shape') and not hasattr(n, '__len__'):
            n = [n]
        return tf.tile(x, n)

    def flatten(self, x):
        return tf.reshape(x, [-1])

    def expand_dims(self, x, axis=-1):
        '''Adds a 1-sized dimension at index "axis".
        '''
        return tf.expand_dims(x, axis)

    def squeeze(self, x, axis):
        '''Removes a 1-dimension from the tensor at index "axis".
        '''
        return tf.squeeze(x, [axis])

    def stack(self, x):
        return tf.pack(x)

    # NN OPERATIONS

    def sigmoid(self, x):
        '''Element-wise sigmoid.
        '''
        return tf.nn.sigmoid(x)

    def hard_sigmoid(self, x):
        '''Segment-wise linear approximation of sigmoid.
        Faster than sigmoid.
        '''
        x = (0.2 * x) + 0.5
        x = self.clip(x, 0, 1.)
        return x

    def tanh(self, x):
        '''Element-wise tanh.
        '''
        return tf.nn.tanh(x)

    def relu(self, x, alpha=0., max_value=None):
        '''Rectified linear unit

        # Arguments
            alpha: slope of negative section.
            max_value: saturation threshold.
        '''
        negative_part = tf.nn.relu(-x)
        x = tf.nn.relu(x)
        if max_value is not None:
            x = self.clip(x, 0, max_value)
        if isinstance(alpha, (tuple, list, np.ndarray)) or np.isscalar(alpha):
            alpha = tf.constant(alpha, dtype=_FLOATX)
        x -= alpha * negative_part
        return x

    def softmax(self, x):
        '''Softmax of a tensor.
        '''
        # TODO tensorflow's implementation is randomized
        return tf.nn.softmax(x)

    def softplus(self, x):
        '''Softplus of a tensor.
        '''
        return tf.nn.softplus(x)

    def softsign(self, x):
        return tf.nn.softsign(x)

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

        train_x = x*random_tensor
        train_x /= retain_prob
        return self.in_train_phase(train_x, x)

    def _to_NHWZC(self, x):
        # (nb_sample, input_depth, rows, cols, ...) ->
        # (nb_sample, rows, cols, ..., input_depth)
        ndim = self.ndim(x)
        assert ndim==4 or ndim==5
        if ndim==4:
            return self.transpose(x, [0,2,3,1])
        elif ndim==5:
            return self.transpose(x, [0,2,3,4,1])

    def _to_NCHWZ(self, x):
        # (nb_sample, rows, cols, ..., input_depth) ->
        # (nb_sample, input_depth, rows, cols, ...)
        ndim = self.ndim(x)
        assert ndim==4 or ndim==5
        if ndim==4:
            return self.transpose(x, [0,3,1,2])
        elif ndim==5:
            return self.transpose(x, [0,4,1,2,3])

    def conv2d(self, x, kernel, strides=(1, 1), input_shape=None, filter_shape=None):
        strides = (1,) + strides + (1,)

        if _FLOATX == 'float64':
            # tf conv2d only supports float32
            x = self.cast(x, 'float32')
            kernel = self.cast(kernel, 'float32')

        # kernel shape: (input_depth, output_depth, k_rows, k_cols)
        # TF kernel shape: (k_rows, k_cols, input_depth, output_depth)
        kernel = self.transpose(kernel, [2, 3, 0, 1])

        x = self._to_NHWZC(x)
        x = tf.nn.conv2d(x, kernel, strides, padding='VALID')
        x = self._to_NCHWZ(x)

        if _FLOATX == 'float64':
            x = self.cast(x, 'float64')
        return x

    def pool(self, x, mode, pool_size, strides, padding='valid'):

        if strides is None:
            strides = pool_size
        assert len(strides)==len(pool_size)

        if len(pool_size)==2:
            pool_fn = tf.nn.max_pool if mode=='max' else tf.nn.avg_pool
        else:
            pool_fn = tf.nn.max_pool3d if mode=='max' else tf.nn.avg_pool3d

        strides = (1,) + strides + (1,)
        pool_size = (1,) + pool_size + (1,)
        x = self._to_NHWZC(x)
        x = pool_fn(x, pool_size, strides, padding=padding.upper())
        return self._to_NCHWZ(x)

    def padding(self, x, pad, pad_dims, output_shape):
        # x shape: (nb_sample, input_depth, rows, cols)
        pattern = [[0, 0], [0, 0]]  # nb_sample, input_depth does not change
        for i in range(2, len(output_shape)):
            if i not in pad_dims:
                pattern.append([0,0])
            else:
                p = pad[i-2]
                if isinstance(p, (tuple,list)):
                    assert len(p)==2
                    pattern.append([p[0], p[1]])
                else:
                    pattern.append([p, p])

        return tf.pad(x, pattern)

    def rnn(self, step_function, inputs, initial_states,
            go_backwards=False, mask=None,
            unroll=False, input_length=None):
        ndim = len(inputs.get_shape())
        assert ndim >= 3, "Input should be at least 3D."
        axes = [1, 0] + list(range(2, ndim))
        inputs = tf.transpose(inputs, (axes))
        input_list = tf.unpack(inputs)

        states = initial_states
        successive_states = []
        successive_outputs = []
        if go_backwards:
            input_list.reverse()

        if mask is not None:
            # Transpose not supported by bool tensor types, hence round-trip to uint8.
            mask = tf.cast(mask, tf.uint8)
            if len(mask.get_shape()) == ndim-1:
                mask = self.expand_dims(mask)
            mask = tf.cast(tf.transpose(mask, axes), tf.bool)
            mask_list = tf.unpack(mask)

            if go_backwards:
                mask_list.reverse()

            for input, mask_t in zip(input_list, mask_list):
                output, new_states = step_function(input, states)

                # tf.select needs its condition tensor to be the same shape as its two
                # result tensors, but in our case the condition (mask) tensor is
                # (nsamples, 1), and A and B are (nsamples, ndimensions). So we need to
                # broadcast the mask to match the shape of A and B. That's what the
                # tile call does, is just repeat the mask along its second dimension
                # ndimensions times.
                tiled_mask_t = tf.tile(mask_t, tf.pack([1, tf.shape(output)[1]]))

                if len(successive_outputs) == 0:
                    prev_output = self.zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = tf.select(tiled_mask_t, output, prev_output)

                return_states = []
                for state, new_state in zip(states, new_states):
                    # (see earlier comment for tile explanation)
                    tiled_mask_t = tf.tile(mask_t, tf.pack([1, tf.shape(new_state)[1]]))
                    return_states.append(tf.select(tiled_mask_t, new_state, state))

                states = return_states
                successive_outputs.append(output)
                successive_states.append(states)
        else:
            for input in input_list:
                output, states = step_function(input, states)
                successive_outputs.append(output)
                successive_states.append(states)

        last_output = successive_outputs[-1]
        outputs = tf.pack(successive_outputs)
        new_states = successive_states[-1]

        axes = [1, 0] + list(range(2, len(outputs.get_shape())))
        outputs = tf.transpose(outputs, axes)
        return last_output, outputs, new_states

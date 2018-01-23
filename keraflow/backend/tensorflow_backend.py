import warnings

import numpy as np
import tensorflow as tf

from ..utils import KeraFlowError as KError
from .common import _EPSILON, _FLOATX, Backend

py_all = all
py_sum = sum


class TensorflowBackend(Backend):

    def __init__(self, **kwargs):
        super(TensorflowBackend, self).__init__(**kwargs)
        # If a default TensorFlow session is available, we will use it (e.g. user use context manager to set the default session).
        # Else, we will create a session and return it.
        if tf.get_default_session() is not None:
            self.session = tf.get_default_session()
        else:
            self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    def set_session(self, session):
        """Sets the TensorFlow session to a customized session.
        @param session: A TF Session.
        """
        self.session = session

    def clear_session(self):
        """Destroys the current TF graph and creates a new one.
        Useful to avoid clutter from old models / layers.
        """
        tf.reset_default_graph()
        # call __init__ to reset learning_phase, uid_affixes
        self.__init__(self._name, _EPSILON, _FLOATX)

    def reset_random_state(self):
        tf.set_random_seed(self._seed)

    # Basic tensor contructor and property
    def constant(self, value, dtype=_FLOATX, shape=None, name=None):
        """Creates a constant tensor.
        @param value: A constant value (or list)
        @param dtype: The type of the elements of the resulting tensor.
        @param shape: Optional dimensions of resulting tensor.
        @param name: Optional name for the tensor.
        @return tensor.
        """
        return tf.constant(value, dtype=dtype, shape=shape, name=name)

    def variable(self, value, dtype=_FLOATX, name=None):
        """Instantiates a variable and returns it.

        @param value: Numpy array, initial value of the tensor.
        @param dtype: Tensor type.
        @param name: Optional name string for the tensor.
        @return tf constant tensor.
        """
        if hasattr(value, 'tocoo'):
            sparse_coo = value.tocoo()
            indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
                                      np.expand_dims(sparse_coo.col, 1)), 1)
            v = tf.SparseTensor(indices=indices,
                                values=sparse_coo.data,
                                dense_shape=sparse_coo.shape)
            return v

        v = tf.Variable(value, dtype=tf.as_dtype(dtype), name=name)
        self.session.run(v.initializer)
        return v

    def placeholder(self, shape=None, ndim=None, dtype=_FLOATX, sparse=False, name=None):
        """Instantiates a placeholder tensor and returns it.

        @param shape: Shape of the placeholder (integer tuple, may include `None` entries).
        @param ndim: Number of axes of the tensor. At least one of {`shape`, `ndim`} must be specified.  If both are specified, `shape` is used.
        @param dtype: Placeholder type.
        @param sparse: Boolean, whether the placeholder should have a sparse type.
        @param name: Optional name string for the placeholder.
        @return Tensor instance.
        """
        if not shape:
            if ndim:
                shape = tuple([None for _ in range(ndim)])
        if sparse:
            x = tf.sparse_placeholder(dtype, shape=shape, name=name)
        else:
            x = tf.placeholder(dtype, shape=shape, name=name)
        return x

    def _to_tensor(self, x, of_type):
        """Convert the input `x` to a tensor of type `dtype`.
        @param x: An object to be converted (numpy array, list, tensors).
        @param dtype: The destination type.
        @return tensor.
        """
        dtype = of_type.dtype.base_dtype
        return tf.convert_to_tensor(x, dtype=dtype)

    def is_sparse(self, tensor):
        """Returns whether a tensor is a sparse tensor.
        @param tensor: A tensor instance.
        @return A boolean.
        """
        return isinstance(tensor, tf.SparseTensor)

    def to_dense(self, tensor):
        """Converts a sparse tensor into a dense tensor and returns it.

        @param tensor: A tensor instance (potentially sparse).
        @return A dense tensor.
        """
        if self.is_sparse(tensor):
            return tf.sparse_tensor_to_dense(tensor)
        else:
            return tensor

    # TENSOR OPERATION

    def shape(self, x):
        '''Returns the symbolic shape of a tensor.
        '''
        return tf.shape(x)

    def int_shape(self, x):
        '''Returns the shape of a tensor as a tuple of integers or None entries.
        Note that this function only works with TensorFlow.
        '''
        try:
            return tuple(x.get_shape().as_list())
        except ValueError:
            return None

    def ndim(self, x):
        '''Returns the number of axes in a tensor, as an integer.
        @param x: Tensor or variable.
        @return int. Number of axes.
        '''
        dims = x.get_shape()._dims
        if dims is not None:
            return len(dims)
        return None

    def dtype(self, x):
        '''Returns the dtype of a tensor, as a string.
        @param x: Tensor or variable.
        @return str. dtype of `x`.
        '''
        return x.dtype.base_dtype.name

    def eval(self, x):
        '''Evaluates the value of a tensor.
        @param x: A variable.
        @return numpy array.
        '''
        return self.to_dense(x).eval(session=self.session)

    def cast(self, x, dtype):
        '''Casts a tensor to a different dtype.
        '''
        return tf.cast(x, dtype)

    def set_value(self, x, value):
        '''Sets the value of a tensor variable, from a Numpy array.
        @param x: tensor. Target tensor.
        @param value: numpy array. Value to set.
        '''
        tf.assign(x, np.asarray(value)).op.run(session=self.session)

    def switch(self, condition, then_expression, else_expression):
        '''Switches between two operations depending on a scalar value. Note that both `then_expression` and `else_expression` should be symbolic tensors of the *same shape*.
        @param condition: scalar tensor (int or bool).
        @param then_expression: TensorFlow operation.
        @param else_expression: TensorFlow operation.
        @return tensor. The selected tensor.
        '''
        if condition.dtype != tf.bool:
            condition = tf.cast(condition, 'bool')
        cond_ndim = self.ndim(condition)
        if not cond_ndim:
            if not callable(then_expression):
                def then_expression_fn():
                    return then_expression
            else:
                then_expression_fn = then_expression
            if not callable(else_expression):
                def else_expression_fn():
                    return else_expression
            else:
                else_expression_fn = else_expression
            x = tf.cond(condition,
                        then_expression_fn,
                        else_expression_fn)
        else:
            # tf.where needs its condition tensor
            # to be the same shape as its two
            # result tensors
            if callable(then_expression):
                then_expression = then_expression()
            if callable(else_expression):
                else_expression = else_expression()
            expr_ndim = self.ndim(then_expression)
            if cond_ndim > expr_ndim:
                raise ValueError('Rank of `condition` should be less than or'
                                 ' equal to rank of `then_expression` and '
                                 '`else_expression`. ndim(condition)=' +
                                 str(cond_ndim) + ', ndim(then_expression)'
                                 '=' + str(expr_ndim))
            if cond_ndim > 1:
                ndim_diff = expr_ndim - cond_ndim
                cond_shape = tf.concat([tf.shape(condition), [1] * ndim_diff], axis=0)
                condition = tf.reshape(condition, cond_shape)
                expr_shape = tf.shape(then_expression)
                shape_diff = expr_shape - cond_shape
                tile_shape = tf.where(shape_diff > 0, expr_shape, tf.ones_like(expr_shape))
                condition = tf.tile(condition, tile_shape)
            x = tf.where(condition, then_expression, else_expression)
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

        @param inputs: list of placeholder/variable tensors.
        @param outputs: list of output tensors.
        @param updates: list of update tuples (old_tensor, new_tensor).
        @return tensorflow function.
        '''
        if len(kwargs) > 0:
            msg = [
                "Expected no kwargs, you passed %s" % len(kwargs),
                "kwargs passed to function are ignored with Tensorflow backend"
            ]
            warnings.warn('\n'.join(msg))
        return self._Function(inputs, outputs, updates=updates)

    def gradients(self, loss, variables):
        '''Returns the gradients of `variables` (list of tensor variables) with regard to `loss`.
        @param loss: Scalar tensor to minimize.
        @param variables: List of variables.
        @return tensor. Gradients tensor.
        '''
        return tf.gradients(loss, variables)

    # UPDATE OPs
    def update(x, new_x):
        """Update the value of tensor `x` to that of `new_x`.
        @param x: tensor. Target tensor.
        @param new_x: tensor. Source tensor, which is of same shape as `x`.
        @return tensor. The tensor `x` updated.
        """
        return tf.assign(x, new_x)

    def update_add(x, increment):
        """Update the value of `x` by adding `increment`.
        @param x: tensor. Target tensor.
        @param increment: tensor. Increment tensor, which is of same shape as `x`.
        @return tensor. The tensor `x` updated.
        """
        return tf.assign_add(x, increment)

    def update_sub(x, decrement):
        """Update the value of `x` by subtracting `decrement`.
        @param x: tensor. Target tensor.
        @param decrement: tensor. Decrement tensor, which is of same shape as `x`.
        @return tensor. The tensor `x` updated.
        """
        return tf.assign_sub(x, decrement)

    # RANDOMNESS

    def random_normal(self, shape, mean=0.0, std=1.0, dtype=_FLOATX):
        return tf.random_normal(shape, mean=mean, stddev=std, dtype=dtype)

    def random_uniform(self, shape, minval=0.0, maxval=1.0, dtype=_FLOATX, seed=None):
        """Returns a tensor with uniform distribution of values.

        # Arguments
            shape: A tuple of integers, the shape of tensor to create.
            minval: A float, lower boundary of the uniform distribution
                to draw samples.
            maxval: A float, upper boundary of the uniform distribution
                to draw samples.
            dtype: String, dtype of returned tensor.
            seed: Integer, random seed.

        # Returns
            A tensor.
        """
        if seed is None:
            seed = np.random.randint(10e6)
        return tf.random_uniform(shape, minval=minval, maxval=maxval,
                                 dtype=dtype, seed=seed)

    def random_binomial(self, shape, p=0.0, dtype=_FLOATX, seed=None):
        """Returns a tensor with random binomial distribution of values.

        # Arguments
            shape: A tuple of integers, the shape of tensor to create.
            p: A float, `0. <= p <= 1`, probability of binomial distribution.
            dtype: String, dtype of returned tensor.
            seed: Integer, random seed.

        # Returns
            A tensor.
        """
        if seed is None:
            seed = np.random.randint(10e6)
        return tf.where(tf.random_uniform(shape, dtype=dtype, seed=seed) <= p,
                        tf.ones(shape, dtype=dtype),
                        tf.zeros(shape, dtype=dtype))

    # NUMPY API

    def zeros(self, shape, dtype=_FLOATX, name=None):
        '''Instantiates an all-zeros tensor variable.
        @param shape: int tuple. Shape of the returned tensor.
        @param dtype: str. data type of the returned tensor.
        @param name: str. Name of the returned tensor.
        @return tensor. Filled with zeros.
        '''
        return self.variable(np.zeros(shape), dtype, name)

    def ones(self, shape, dtype=_FLOATX, name=None):
        '''Instantiates an all-ones tensor variable.
        @param shape: int tuple. Shape of the returned tensor.
        @param dtype: str. data type of the returned tensor.
        @param name: str. Name of the returned tensor.
        @return tensor. Filled with ones.
        '''
        return self.variable(np.ones(shape), dtype, name)

    def eye(self, size, dtype=_FLOATX, name=None):
        '''Instantiate an identity matrix.
        @param size: int. Number of rows/columns.
        @param dtype: str. data type of the returned tensor.
        @param name: str. Name of the returned tensor.
        @return tensor. Identity matrix.
        '''
        return self.variable(np.eye(size), dtype, name)

    def zeros_like(self, x, dtype=None, name=None):
        '''Instantiates an all-zeros tensor of the same shape as another tensor.

        @param x: tensor.
        @param dtype: String, dtype of returned tensor. None uses the dtype of x.
        @param name: String, name for the tensor to create.
        @return tensor. With the shape of x filled with zeros.
        '''
        # TODO not a Variable but a Tensor
        return tf.zeros_like(x, name=name)

    def ones_like(self, x, name=None):
        '''Instantiates an all-ones tensor of the same shape as another tensor.

        @param x: tensor.
        @param dtype: String, dtype of returned tensor. None uses the dtype of x.
        @param name: String, name for the tensor to create.
        @return tensor. With the shape of x filled with ones.
        '''
        # TODO not a Variable but a Tensor
        return tf.ones_like(x, name=name)

    def identity(x, name=None):
        """Returns a tensor with the same content as the input tensor.

        @param x: tensor. The input tensor.
        @param name: str. Name for the variable to create.
        @return tensor.

        @return tensor. A tensor of the same shape, type and content.
        """
        # TODO not a Variable but a Tensor
        return tf.identity(x, name)

    def dot(self, x, y):
        '''numpy.dot on tensors
        @param x: tensor.
        @param y: tensor.
        @return tensor. Dot product of `x` and `y`.
        '''
        def get_shape_tuple(tensor):
            res = []
            for i, s in zip(self.int_shape(tensor), tf.unstack(tf.shape(tensor))):
                if i is not None:
                    res.append(i)
                else:
                    res.append(s)
            return tuple(res)

        if self.ndim(x) is not None and (self.ndim(x) > 2 or self.ndim(y) > 2):
            x_shape = get_shape_tuple(x)
            y_shape = get_shape_tuple(y)
            y_permute_dim = list(range(self.ndim(y)))
            y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
            xt = tf.reshape(x, [-1, x_shape[-1]])
            yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
            return tf.reshape(tf.matmul(xt, yt),
                              x_shape[:-1] + y_shape[:-2] + y_shape[-1:])

        if self.is_sparse(x):
            out = tf.sparse_tensor_dense_matmul(x, y)
        else:
            out = tf.matmul(x, y)
        return out

    def gather(self, reference, indices):
        '''Retrieves the vectors of indices `indices` in the 2D tensor `reference`.
        @param reference: a 2D tensor.
        @param indices: 2D int tensor or list.
        @return tensor. 3D tensor.
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
            rank = self.ndim(tensors[0])
            if rank:
                axis %= rank
            else:
                axis = 0

        if py_all([self.is_sparse(x) for x in tensors]):
            return tf.sparse_concat(axis, tensors)
        else:
            return tf.concat([self.to_dense(x) for x in tensors], axis)

    def reshape(self, x, shape):
        '''Reshapes a tensor to the specified shape.
        '''
        return tf.reshape(x, shape)

    def transpose(self, x, axes=None):
        '''np.transpose on tensor. Permute the dimensions of a tensor.
        @param x: tensor.
        @param axes: tuple. New dimension indices, e.g. `(0, 2, 1)`.
        @return tensor.
        '''
        return tf.transpose(x, perm=axes)

    def repeat(self, x, rep, axis):
        """np.repeat on tensor. Repeat elements of an tensor along axis rep times.
        @param x: tensor.
        @param rep: int. Number of times to repeat.
        @param axis: int. Axis along which to repeat.
        @return tensor.
        """
        x_shape = x.get_shape().as_list()
        # For static axis
        if x_shape[axis] is not None:
            # slices along the repeat axis
            splits = tf.split(value=x, num_or_size_splits=x_shape[axis], axis=axis)
            # repeat each slice the given number of reps
            x_rep = [s for s in splits for _ in range(rep)]
            return self.concatenate(x_rep, axis)

        # Here we use tf.tile to mimic behavior of np.repeat so that
        # we can handle dynamic shapes (that include None).
        # To do that, we need an auxiliary axis to repeat elements along
        # it and then merge them along the desired axis.

        # Repeating
        auxiliary_axis = axis + 1
        x_shape = tf.shape(x)
        x_rep = tf.expand_dims(x, axis=auxiliary_axis)
        reps = np.ones(len(x.get_shape()) + 1)
        reps[auxiliary_axis] = rep
        x_rep = tf.tile(x_rep, reps)

        # Merging
        reps = np.delete(reps, auxiliary_axis)
        reps[axis] = rep
        reps = tf.constant(reps, dtype='int32')
        x_shape = x_shape * reps
        x_rep = tf.reshape(x_rep, x_shape)

        # Fix shape representation
        x_shape = x.get_shape().as_list()
        x_rep.set_shape(x_shape)
        return x_rep

    def tile(self, x, n):
        """np.tile on tensor. Construct a tensor by repeating x the number of times given by reps.
        @param x: tensor.
        @param n: list of int. Specifies repeat for each dimension. When n is an int, it specifies repeat for the last dimension.
        @return tensor.
        """
        if isinstance(n, int):
            n = [n]
        return tf.tile(x, n)

    def flatten(self, x):
        return self.reshape(x, [-1])

    def batch_flatten(self, x):
        """batch version of flatten. Flattens each data samples of a batch.
        @param x: tensor.
        @return tensor.
        """
        x = self.reshape(x, self.stack([-1, self.prod(self.shape(x)[1:])]))
        return x

    def expand_dims(self, x, axis=-1):
        '''np.expand_dims on tensor. Insert a new axis that will appear at the `axis` position in the expanded tensor shape.

        @param x: tensor.
        @param axis: int. Position where to add a new axis.
        @return tensor.
        '''
        return tf.expand_dims(x, axis)

    def squeeze(self, x, axis):
        '''np.squeeze on tensor. Remove single-dimensional entries from the shape of a tensor.
        @param x: A tensor or variable.
        @param axis: int. Axis to drop.
        @return tensor.
        '''
        return tf.squeeze(x, [axis])

    def stack(self, x, axis=0):
        """np.stack on tensor. Join a sequence of tensors along a new axis.
        @param x: List of tensors.
        @param axis: Axis along which to perform stacking.
        @return tensor. The output's dimension equals to the dimension of `x` plus 1.
"""
        return tf.stack(x, axis=axis)

    # NN OPERATIONS

    def relu(self, x, alpha=0., max_value=None):
        '''Rectified linear unit. With default values, it returns element-wise `max(x, 0)`.
        @param x: tensor.
        @param alpha: A scalar, slope of negative section (default=`0.`).
        @param max_value: Saturation threshold.
        @return tensor.
        '''
        if alpha != 0.:
            x = tf.nn.leaky_relu(x, alpha)
        else:
            x = tf.nn.relu(x)

        if max_value is not None:
            x = tf.minimum(x, tf.convert_to_tensor(max_value, dtype=_FLOATX))
        return x

    def elu(x, alpha=1.):
        """Exponential linear unit.
        @param x: tensor.
        @param alpha: A scalar, slope of negative section.
        @return tensor.
        """
        res = tf.nn.elu(x)
        if alpha == 1:
            return res
        else:
            return tf.where(x > 0, res, alpha * res)

    def softmax(self, x):
        '''Softmax of a tensor.
        @param x: tensor.
        @return tensor.
        '''
        # TODO tensorflow's implementation is randomized
        return tf.nn.softmax(x)

    def softplus(self, x):
        '''Softplus of a tensor.
        @param x: tensor.
        @return tensor.
        '''
        return tf.nn.softplus(x)

    def softsign(self, x):
        '''Softsign of a tensor.
        @param x: tensor.
        @return tensor.
        '''
        return tf.nn.softsign(x)

    def sigmoid(self, x):
        '''Element-wise sigmoid.
        @param x: tensor.
        @return tensor.
        '''
        return tf.nn.sigmoid(x)

    def hard_sigmoid(self, x):
        '''Segment-wise linear approximation of sigmoid. Faster than sigmoid.
        Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`. In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.
        @param x: tensor.
        @return tensor.
        '''
        x = (0.2 * x) + 0.5
        x = self.clip(x, 0, 1.)
        return x

    def tanh(self, x):
        '''Element-wise tanh.
        '''
        return tf.nn.tanh(x)

    def dropout(self, x, drop_rate, noise_shape=None, seed=None):
        '''Sets entries in `x` to zero at random, while scaling the entire tensor.
        @param x: tensor
        @param drop_rate: fraction of the entries in the tensor that will be set to 0.
        @param noise_shape: shape for randomly generated keep/drop flags, must be broadcastable to the shape of `x`
        '''
        assert drop_rate > 0. or drop_rate < 1, 'Dropout drop_rate must be in interval [0, 1].'

        retain_prob = 1. - drop_rate
        if seed is None:
            seed = np.random.randint(10e6)

        if noise_shape is None:
            random_tensor = self.random_binomial(self.shape(x), p=retain_prob)
        else:
            random_tensor = self.random_binomial(noise_shape, p=retain_prob)

        train_x = x*random_tensor
        train_x /= retain_prob
        return self.in_train_phase(train_x, x)
        # # the dummy 1. works around a TF bug
        # # (float32_ref vs. float32 incompatibility)
        # return tf.nn.dropout(x * 1., retain_prob, noise_shape, seed=seed)

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
        if ndim < 3:
            raise ValueError('Input should be at least 3D.')

        # Transpose to time-major, i.e.
        # from (batch, time, ...) to (time, batch, ...)
        axes = [1, 0] + list(range(2, ndim))
        inputs = tf.transpose(inputs, (axes))
        input_list = tf.unstack(inputs)

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
            mask_list = tf.unstack(mask)

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
                tiled_mask_t = tf.tile(mask_t, tf.stack([1, tf.shape(output)[1]]))

                if len(successive_outputs) == 0:
                    prev_output = self.zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = tf.where(tiled_mask_t, output, prev_output)

                return_states = []
                for state, new_state in zip(states, new_states):
                    # (see earlier comment for tile explanation)
                    tiled_mask_t = tf.tile(mask_t, tf.stack([1, tf.shape(new_state)[1]]))
                    return_states.append(tf.where(tiled_mask_t, new_state, state))

                states = return_states
                successive_outputs.append(output)
                successive_states.append(states)
        else:
            for input in input_list:
                output, states = step_function(input, states)
                successive_outputs.append(output)
                successive_states.append(states)

        last_output = successive_outputs[-1]
        outputs = tf.stack(successive_outputs)
        new_states = successive_states[-1]

        axes = [1, 0] + list(range(2, len(outputs.get_shape())))
        outputs = tf.transpose(outputs, axes)
        return last_output, outputs, new_states

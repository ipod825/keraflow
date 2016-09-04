import numpy as np

from .. import backend as B
from .. import utils
from ..utils import KeraFlowError as KError
from .base import Layer, MultiInputLayer


class ExpandDims(Layer):
    '''Expand dimension of the input tensor. Behaves like numpy.expand_dims().
    - input_shape: nD, `(nb_samples, x, | y, ...)
    - output_shape: (n+1)D, `(nb_samples, x, 1, y, ...)`
    '''
    def __init__(self, axis, include_batch_dim=False, **kwargs):
        '''
        @param axis: int. The axis to expand dimension.
        @param include_batch_dim: boolean. If true, `axis`=0 refers to the first dimension; otherwise, `axis`=0 refers to the second dimension(0->1, 1->2...).
        @param kwargs: see Layer.__init__.
        '''
        super(ExpandDims, self).__init__(**kwargs)
        self.axis = axis
        self.include_batch_dim = include_batch_dim
        self._axis = axis
        if not include_batch_dim:
            self._axis += 1

    def output_shape(self, input_shape):
        res = list(input_shape)
        res.insert(self._axis, 1)
        return res

    def output(self, x):
        return B.expand_dims(x, axis=self._axis)


class PermuteDims(Layer):
    '''Permutes the dimensions of the input tensor according to a given pattern. Behaves like numpy.transpose().
    - input_shape: nD, `(nb_samples, x, y, ...)`
    - output_shape: nD, `(nb_samples, y, x, ...)`
    '''
    def __init__(self, dims, include_batch_dim=False, **kwargs):
        '''
        @param dims: list of int. The permuting dimension pattern.
        @param include_batch_dim: boolean. If true, in `dims`, 0 refers to the first dimension; otherwise, 0 refers to the second dimension(0->1, 1->2...).
        @param kwargs: see Layer.__init__.
        '''
        super(PermuteDims, self).__init__(**kwargs)
        self.dims = dims
        self.include_batch_dim = include_batch_dim

    def output_shape(self, input_shape):
        if self.include_batch_dim:
            return tuple(np.asarray(input_shape)[self.dims])
        else:
            return (input_shape[0],) + tuple(np.asarray(input_shape[1:])[self.dims])

    def output(self, x):
        if self.include_batch_dim:
            return B.transpose(x, dims=self.dims)
        else:
            dims = np.asarray(self.dims)+1
            return B.transpose(x, dims=[0,]+list(dims))


class Reshape(Layer):
    '''Reshapes the input tensor according to a given pattern. Behaves like numpy.reshape().
    - input_shape: nD, `(nb_samples, x, y, ...)`
    - output_shape: ?D, `(nb_samples, a*y, ...)`
    '''
    def __init__(self, target_shape, include_batch_dim=False, **kwargs):
        '''
        @param target_shape: list of int. Target shape. You could specify at most one -1, which will be automatically determined.
        @param include_batch_dim: boolean. When False, in `target_shape`, 0 refers to the second dimension of the tensor, i.e. the dimension next to the batch size dimension. Otherwise, 0 refers to the first (batch) dimension.
        @param kwargs: see Layer.__init__.
        '''
        super(Reshape, self).__init__(**kwargs)
        self.unknown_dimension = self._calc_unkonwn_dimension(target_shape)
        self.target_shape = utils.to_tuple(target_shape)
        self.include_batch_dim = include_batch_dim

    def _calc_unkonwn_dimension(self, target_shape):
        count = 0
        unknown_dimension = None
        for d, s in enumerate(target_shape):
            if s<0 and s!=-1:
                raise KError("Each dimension should be >0 or ==-1!!")
            if s==-1:
                count+=1
                unknown_dimension = d
        if count>1:
            raise KError("Target shape should contain only one undetermined dimension.")
        return unknown_dimension

    def output_shape(self, input_shape):
        def calc_target_shape(input_shape):
            inps = [s for s in input_shape if s is not None]
            input_total = np.prod(inps, dtype=int)
            target_total = np.prod(self.target_shape, dtype=int)
            if input_total % target_total !=0:
                raise KError("Total size of new array must be unchanged!! Input shape:{}. Target shape:{}".format(input_shape, self.target_shape))

            res = list(self.target_shape)
            if self.unknown_dimension is not None:
                res[self.unknown_dimension] = int(input_total / (-target_total))

            return res

        if self.include_batch_dim:
            target_shape = calc_target_shape(input_shape)
            if input_shape[0] is None:
                if self.target_shape[0]!=-1:
                    raise KError("Trying to change batch size while original batch size is undetermined. This might cause potential error. Please specify the original batch via Input's argument.")
                target_shape[0] = None
            return target_shape
        else:
            batch_size, input_shape = input_shape[0], input_shape[1:]
            target_shape = calc_target_shape(input_shape)
            return [batch_size,] + target_shape

    def output(self, x):
        input_shape = self.get_tensor_shape(x)
        output_shape = list(self.output_shape(input_shape))
        if output_shape[0] is None:
            output_shape[0] = -1
        return B.reshape(x, output_shape)


class Flatten(Layer):
    '''Flatten the input tensor into 1D. Behaves like numpy.reshape().
    - input_shape: nD, `(nb_samples, x, y, ...)`
    - output_shape: 2D, `(nb_samples, Prod(x,y,...))`
    '''
    def __init__(self, include_batch_dim=False, **kwargs):
        '''
        @param include_batch_dim: boolean. If False, batch dimension is kept (i.e the output shape will be of two dimensions).
        @param kwargs: see Layer.__init__.
        '''
        super(Flatten, self).__init__(**kwargs)
        self.include_batch_dim = include_batch_dim

    def output_shape(self, input_shape):
        batch_size, input_shape = input_shape[0], input_shape[1:]
        if not all(input_shape):
            raise KError('The shape of the input to "Flatten" is not fully defined. Please do not use "None" for "shape" argument of "Input" layers.')
        if self.include_batch_dim:
            if batch_size is None:
                raise KError('The batch size to "Flatten" is not determined. Please specify "batch_size" argument of "Input" layers.')
            return (np.prod(input_shape),)
        else:
            return (batch_size, np.prod(input_shape))

    def output(self, x):
        if self.include_batch_dim:
            return B.reshape(x, [-1])
        else:
            return B.reshape(x, [-1, B.prod(B.shape(x)[1:])])


class Repeat(Layer):
    '''Repeat the input tensor n times along given axis. Behaves like numpy.reshape().
    - input_shape: nD, `(nb_samples, | x, y, ...)`
    - output_shape: nD, `(nb_samples, n*x, y, ...)`
    '''
    def __init__(self, n, axis=0, include_batch_dim=False, **kwargs):
        '''
        @param n: int. Times to repeat along `axis`.
        @param axis: int. The axis to expand dimension.
        @param include_batch_dim: boolean. If true, `axis`=0 refers to the first dimension; otherwise, `axis`=0 refers to the second dimension(0->1, 1->2...).
        @param kwargs: see Layer.__init__.
        '''
        super(Repeat, self).__init__(**kwargs)
        self.n = n
        self.axis = axis
        self.include_batch_dim = include_batch_dim
        self._axis = axis
        if not include_batch_dim:
            self._axis += 1

    def output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self._axis]*=self.n
        return output_shape

    def output(self, x):
        return B.repeat(x, self.n, self._axis)


class Concatenate(MultiInputLayer):
    '''Concatenate multiple input tensors. Behaves like numpy.concatenate().
    - Accepts multiple input tensors
    - input_shapes: nD, `(nb_samples, x, y1, ...)`, `(nb_samples, x, y2, ...)`
    - output_shape: nD, `(nb_samples, x, y1+y2, ...)`
    '''
    def __init__(self, axis=0, include_batch_dim=False, **kwargs):
        '''
        @param axis: int. The axis to concatenate the input tensors.
        @param include_batch_dim: boolean. If true, `axis`=0 refers to the first dimension; otherwise, `axis`=0 refers to the second dimension(0->1, 1->2...).
        @param kwargs: see Layer.__init__.
        '''
        super(Concatenate, self).__init__(**kwargs)
        self.axis = axis
        self.include_batch_dim = include_batch_dim
        self._axis = axis
        if not include_batch_dim:
            self._axis += 1

    def check_input_shape(self, input_shapes):
        unmatch =False
        shape0 = input_shapes[0]
        for shape in input_shapes[1:]:
            if len(shape0) != len(shape):
                unmatch =True
            for i, (s0, s) in enumerate(zip(shape0, shape)):
                if i!=self._axis and s0 is not None and s0 != s:
                    unmatch = True
                    break
            if unmatch:
                raise KError("Incrroect input shape dimension for {}. Expecting all equal shapes (except axis {}). Given: {}".format(self.name, self.axis, input_shapes))  # cover

    def output_shape(self, input_shapes):
        res = list(input_shapes[0])
        res[self._axis] = sum([shape[self._axis] for shape in input_shapes])
        return res

    def output(self, tensors):
        tensors = utils.to_list(tensors)
        return B.concatenate(tensors, axis=self._axis)


class Lambda(Layer):
    '''Wrapper for implementating simple inline layer.
    - input_shapes: nD, `(nb_samples, x, y, ...)`
    - output_shape: nD, specified by `output_shape` argument.

    # Examples

    ```python
        # add a x -> x^2 layer
        model.add(Lambda(lambda x, factor: x ** 2))
    ```
    '''
    def __init__(self, output_fn, output_shape_fn=None, arguments={}, **kwargs):
        '''
        @param output_fn: The function to transform input tensor into output tensor. The function should take 1 position argument (the input tensor) and variable numbers of keyword argument whose value are specified by `arguments` argument.
        @param output_shape_fn: tuple, list, or (lambda) function. Expected output shape.
        1. If a tuple or list, it should specifies all dimension excepts for the batch dimension.
        2. If a function, it should accept the `input_shape` (tuple) and return an `output_shape` (tuple), such that both `input_shape` and `output_shape` includes the batch dimension.
        @param arguments: dict. Optional keyword arguments to be passed to `output_fn`.
        @param kwargs: see Layer.__init__.
        '''
        super(Lambda, self).__init__(**kwargs)
        self.output_fn = output_fn
        self.output_shape_fn = output_shape_fn
        self.arguments = arguments

    def output_shape(self, input_shape):
        if self.output_shape_fn is None:
            return input_shape
        elif type(self.output_shape_fn) in {tuple, list}:
            return (input_shape[0],) + tuple(self.output_shape_fn)
        else:
            if not hasattr(self.output_shape_fn, '__call__'):
                raise KError('"output_shape_fn" should be a list, a tuple, or a function.')
            shape = self.output_shape_fn(input_shape)
            if not type(shape) in {list, tuple}:
                raise KError('"output_shape_fn" should return a list or a tuple.')
            return shape

    def output(self, x):
        return self.output_fn(x, **self.arguments)


class Activation(Layer):
    '''Applies an activation function to an output.
    - input_shape: nD, `(nb_samples, ...)`
    - output_shape: nD, `(nb_samples, ...)`
    '''
    def __init__(self, activation, **kwargs):
        '''
        @param activation: str/function. Activation function applied on the output. See @ref Activations.md.
        @param kwargs: see Layer.__init__.
        '''
        super(Activation, self).__init__(**kwargs)
        self.activation = utils.get_from_module('activations', activation)

    def output(self, x):
        return self.activation(x)


class ElementWiseSum(MultiInputLayer):
    '''Reduce multiple input tensors by conducting summation operation.
    - Accepts multiple input tensors
    - input_shape: nD, `(nb_samples, ...)`
    - output_shape: nD, `(nb_samples, ...)`
    '''
    def __init__(self, **kwargs):
        '''
        @param kwargs: see Layer.__init__.
        '''
        super(ElementWiseSum, self).__init__(**kwargs)

    def output(self, tensors):
        tensors = utils.to_list(tensors)
        return sum(tensors)


class ElementWiseMult(MultiInputLayer):
    '''Reduce multiple input tensors by conducting multiplication operation.
    - Accepts multiple input tensors
    - input_shape: nD, `(nb_samples, ...)`
    - output_shape: nD, `(nb_samples, ...)`
    '''
    def __init__(self, **kwargs):
        '''
        @param kwargs: see Layer.__init__.
        '''
        super(ElementWiseMult, self).__init__(**kwargs)

    def output(self, tensors):
        tensors = utils.to_list(tensors)
        res = tensors[0]
        for tensor in tensors[1:]:
            res *= tensor
        return res


class Dropout(Layer):
    '''Applies Dropout to the input.
    In training phase, for each unit in the input tensor, Dropout sets it to 0 with probability `drop_rate` and sets it to 1/(1-`drop_rate`) times with probability 1-`drop_rate`.
    In testing phase, Dropout simply returns the input tensor.
    - input_shape: nD, `(nb_samples, ...)`
    - output_shape: nD, `(nb_samples, ...)`
    - Reference: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    '''
    def __init__(self, drop_rate, **kwargs):
        '''
        @param drop_rate: float between 0 and 1. Fraction of the input units to drop.
        @param kwargs: see Layer.__init__.
        '''
        if drop_rate < 0. or drop_rate >= 1:
            raise KError('drop_rate must be in interval [0, 1).')
        super(Dropout, self).__init__(**kwargs)
        self.drop_rate = drop_rate

    def _get_noise_shape(self, x):
        return None

    def output(self, x):
        return B.dropout(x, self.drop_rate, self._get_noise_shape(x))


class Dense(Layer):
    '''Fully connected layer.
    - input_shape: 2D, `(nb_samples, input_dim)`
    - output_shape: 2D, `(nb_samples, output_dim)`
    - parameters:
        - W: `(input_dim, output_dim)`
        - b: `(output_dim,)`
    '''
    def __init__(self, output_dim, init='glorot_uniform', activation='linear', bias=True, **kwargs):
        '''
        @param output_dim: int > 0.
        @param init: str/function. Function to initialize trainable parameters. See @ref Initializations.md.
        @param activation: str/function. Activation function applied on the output. See @ref Activations.md.
        @param bias: boolean. Whether to include a bias (i.e. make the layer affine rather than linear).
        @param kwargs: see Layer.__init__.
        '''
        super(Dense, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.init = utils.get_from_module('initializations', init)
        self.activation = utils.get_from_module('activations', activation)
        self.bias = bias

    def input_dimension(self):
        return 2

    def init_param(self, input_shape):
        input_dim = input_shape[1]

        W = self.init((input_dim, self.output_dim), name='{}_W'.format(self.name))
        if self.bias:
            b = B.zeros((self.output_dim,), name='{}_b'.format(self.name))
            self.set_trainable_params('W', W, 'b', b)
        else:
            self.set_trainable_params('W', W)

    def output(self, x):
        output = B.dot(x, self.W)
        if self.bias:
            output += self.b
        return self.activation(output)

    def output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class Highway(Layer):
    '''Densely connected highway network. A natural extension of LSTMs to feedforward networks.
    - input_shape: 2D, `(nb_samples, input_dim)`
    - output_shape: 2D, `(nb_samples, input_dim)`
    - parameters:
        - W, W_carry: `(input_dim, input_dim)`
        - b, b_carry: `(input_dim,)`
    - Reference: [Highway Networks](http://arxiv.org/pdf/1505.00387v2.pdf)
    '''
    def __init__(self, transform_bias=-2, init='glorot_uniform', activation='linear', bias=True, **kwargs):
        '''
        @param transform_bias: initial value for all elements of `b_carry`.
        @param init: str/function. Function to initialize trainable parameters. See @ref Initializations.md.
        @param activation: str/function. Activation function applied on the output. See @ref Activations.md.
        @param bias: boolean. Whether to include a bias (i.e. make the layer affine rather than linear).
        @param kwargs: see Layer.__init__.
        '''
        super(Highway, self).__init__(**kwargs)
        self.transform_bias = transform_bias
        self.init = utils.get_from_module('initializations', init)
        self.activation = utils.get_from_module('activations', activation)
        self.bias = bias

    def init_param(self, input_shape):
        input_dim = input_shape[1]
        W = self.init((input_dim, input_dim), name='{}_W'.format(self.name))
        W_carry = self.init((input_dim, input_dim), name='{}_W_carry'.format(self.name))

        if self.bias:
            b = B.zeros((input_dim,), name='{}_b'.format(self.name))
            b_carry = B.variable(np.ones((input_dim,)) * self.transform_bias, name='{}_b_carry'.format(self.name))
            self.set_trainable_params('W', W, 'W_carry', W_carry, 'b', b, 'b_carry', b_carry)
        else:
            self.set_trainable_params('W', W, 'W_carry', W_carry)

    def output(self, x):
        y = B.dot(x, self.W_carry)
        if self.bias:
            y += self.b_carry
        transform_weight = utils.get_from_module('activations', 'sigmoid')(y)

        output = B.dot(x, self.W)
        if self.bias:
            output += self.b
        output= self.activation(output)

        return output*transform_weight+(1-transform_weight)*x

    def input_dimension(self):
        return 2

# TODO
# class SpatialDropout2D(Dropout):
#     '''This version performs the same function as Dropout, however it drops entire 2D feature maps instead of individual elements. If adjacent pixels within feature maps are strongly correlated (as is normally the case in early convolution layers) then regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease. In this case, SpatialDropout2D will help promote independence between feature maps and should be used instead.
#     - input_shape: 4D, `(nb_samples, input_depth, input_row, input_col)`
#     - output_shape: 4D, `(nb_samples, input_depth, input_row, input_col)`
#     - Reference: [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/pdf/1411.4280.pdf)
#     '''
#     def __init__(self, drop_rate, **kwargs):
#         '''
#         @param p: float between 0 and 1. Fraction of the input units to drop.
#         '''
#         super(SpatialDropout2D, self).__init__(drop_rate, **kwargs)
#
#     def _get_noise_shape(self, x):
#         input_shape = B.shape(x)
#         return (input_shape[0], input_shape[1], 1, 1)
#
# TODO
# class SpatialDropout3D(Dropout):
#     '''A 3D counterpart of SpatialDropout2D.
#     - input_shape: 5D, `(nb_samples, input_depth, input_x, input_y, input_z)`
#     - output_shape: 5D, `(nb_samples, input_depth, input_x+x_pad, input_y+y_pad, input_z+z_pad)`
#     - Reference: [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/pdf/1411.4280.pdf)
#     '''
#     def __init__(self, drop_rate, **kwargs):
#         '''
#         @param p: float between 0 and 1. Fraction of the input units to drop.
#         '''
#         super(SpatialDropout2D, self).__init__(drop_rate, **kwargs)
#
#     def _get_noise_shape(self, x):
#         input_shape = B.shape(x)
#         return (input_shape[0], input_shape[1], 1, 1, 1)

import math

from .. import backend as B
from .. import utils
from ..utils import KeraFlowError as KError
from .base import Layer
from .core import ExpandDims


def calc_output_shape(pad, input_shape, kernel_shape, strides):
    strides = utils.to_tuple(strides)
    assert len(input_shape)==len(kernel_shape)==len(strides)
    if pad=='same':
        return [int(math.ceil(float(il)/s)) for il,s in zip(input_shape, strides)]
    elif pad=='valid':
        return [int(math.ceil(float(il-k+1)/s)) for il,k,s in zip(input_shape, kernel_shape, strides)]


def calc_pad(pad, input_shape, kernel_shape, strides):
    input_shape = utils.to_tuple(input_shape)
    kernel_shape = utils.to_tuple(kernel_shape)
    strides = utils.to_tuple(strides)
    assert len(input_shape)==len(kernel_shape)==len(strides)
    output_shape = calc_output_shape(pad, input_shape, kernel_shape, strides)
    if pad == 'valid':
        return (0,)*len(input_shape)
    elif pad == 'same':
        pad = []
        all_zero = True
        for il, ol, k, s in zip(input_shape, output_shape, kernel_shape, strides):
            p = float((ol- 1) * s + k - il)
            if p!=0:
                all_zero = False

            if p % 2 == 0:
                pad.append(int(p/2))
            else:
                pad.append((int(p/2), int(p/2)+1))
        if all_zero:
            return (0,)*len(input_shape)
        else:
            return tuple(pad)


class ConvolutionBase(Layer):
    '''Base layer for convolution layers. Do not use this layer in your code.
    '''

    def __init__(self, kernel_shape, strides, pad='valid', bias=True, init='glorot_uniform', activation='linear', **kwargs):
        '''
        @param kernel_shape: tuple of int. Shape of the kernel in the pattern `(nb_kernel, k_rows, k_cols ...)`.
        @param strides: tuple of int. Steps for sliding each kernel for convolution.
        @param pad: str, `'valid'` of `'same'`. See descriptions below.
        @param bias: boolean. Whether to include a bias (i.e. make the layer affine rather than linear).
        @param init: str/function. Function to initialize trainable parameters. See @ref Initializations.md.
        @param activation: str/function. Activation function applied on the output. See @ref Activations.md.
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).

        @note
        We follow tensorflow's padding strategy explained [here](https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html#convolution).
        1. When `pad='same'`, the output length (for each dimension) is computed as:
        ~~~{.py}
        output_length = ceil( float(input_length) / stride )
        ~~~
        And the padding (for each dimension) is computed as:
        ~~~{.py}
        padding = (output_length - 1) * stride + kernel_length - input_length
        pad_front = floor(padding / 2)
        pad_back = (padding % 2==0)? floor(padding / 2): floor(padding / 2) + 1
        ~~~
        When `stride=1`, `output_length` will be equal to `input_length`, which is the reason for the name `same`.
        2. When `pad='valid'`, the output length (for each dimension) is computed as:
        ~~~{.py}
        output_length = ceil(float(input_length-kernel_length+1)/stride)
        ~~~
        And the padding is always 0.
        When `stride=1`, `output_length` will be equal to `input_length-1`.
        '''
        super(ConvolutionBase, self).__init__(**kwargs)
        assert pad in {'valid', 'same'}, 'pad must be in {valid, same}'

        self.kernel_shape = kernel_shape
        self.strides = utils.to_tuple(strides)
        self.pad = pad
        self.bias = bias
        self.init = utils.get_from_module('initializations', init)
        self.activation = utils.get_from_module('activations', activation)

    def init_param(self, input_shape):
        # input shape:  (nb_sample, input_depth, k_rows, k_cols ...)
        # kernel shape: (output_depth, k_rows, k_cols ...)
        # W shape:      (input_depth, output_depth, k_rows, k_cols ...)
        W = self.init((input_shape[1],)+self.kernel_shape, name='{}_W'.format(self.name))
        if self.bias:
            b = B.zeros(self.kernel_shape[0], name='{}_b'.format(self.name))
            self.set_trainable_params('W', W, 'b', b)
        else:
            self.set_trainable_params('W', W)

    def output_shape(self, input_shape):
        # input shape: (nb_sample, input_depth, rows, cols...)
        # kernel shape: (output_depth, k_rows, k_cols...)
        # output shape: (nb_sample, output_depth, new_rows, new_cols...)
        output_shape = calc_output_shape(pad=self.pad,
                                         input_shape=input_shape[2:],
                                         kernel_shape=self.kernel_shape[1:],
                                         strides=self.strides)
        return (input_shape[0], self.kernel_shape[0]) + tuple(output_shape)

    def _pad_x(self, x):
        raise NotImplementedError

    def output(self, x):
        input_shape = self.get_tensor_shape(x)
        pad = calc_pad(pad=self.pad,
                       input_shape=input_shape[2:],
                       kernel_shape=self.kernel_shape[1:],
                       strides=self.strides)

        x = self._pad_x(x, pad)
        input_shape = self.get_tensor_shape(x)
        filter_shape = (input_shape[1],)+self.kernel_shape

        output = B.conv2d(x, self.W,
                          strides=self.strides,
                          input_shape=input_shape,
                          filter_shape=filter_shape)
        if self.bias:
            output += B.reshape(self.b, (self.kernel_shape[0],)+(1,)*(len(self.kernel_shape)-1))
        output = self.activation(output)
        return output


class Convolution1D(ConvolutionBase):
    '''Convolution layer for convolving (sequence_length, input_dim) inputs.
    - input_shape: 3D, `(nb_samples, sequence_length, input_dim)`
    - output_shape: 3D, `(nb_samples, output_sequence_length, nb_kernel)`
    - parameters:
        - W: `(1, 1, kernel_row, input_col)`
        - b: `(nb_kernel,)`

    @note
    1. `output_sequence_length` is determined by `pad` and `stride`. For details, please see ConvolutionBase.
    2. The shape of `W` has two additional dimensions (the 1s) due to implementation issue. Be aware of this when initializing the layer with `initial_weights` argument.
    '''
    def __init__(self, nb_kernel, kernel_row, stride=1, pad='valid', bias=True, init='uniform', activation='linear', **kwargs):
        '''
        @param nb_kernel: int. Number of convolution kernels to use.
        @param kernel_row: int. The height of the each kernel. The width of kernel will always be the input's width.
        @param stride: int. Step for vertically sliding each kernel for convolution.
        @param pad: str, 'valid' of 'same'. See ConvolutionBase
        @param bias: boolean. Whether to include a bias (i.e. make the layer affine rather than linear).
        @param init: str/function. Function to initialize trainable parameters. See @ref Initializations.md.
        @param activation: str/function. Activation function applied on the output. See @ref Activations.md.
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).
        '''
        self.nb_kernel = nb_kernel
        self.kernel_row = kernel_row
        self.stride = stride
        super(Convolution1D, self).__init__(kernel_shape=(nb_kernel, kernel_row, None),
                                            strides=(stride, 1),
                                            pad=pad,
                                            bias=bias,
                                            init=init,
                                            activation=activation,
                                            **kwargs)

    def init_param(self, input_shape):
        # input shape: (nb_sample, rows, cols)
        # 2d input shape: (nb_sample, input_depth, rows, cols)
        # kernel shape: (output_depth, k_rows, None)
        # 2d kernl shape: (input_depth, k_rows, cols)

        self.kernel_shape = self.kernel_shape[:-1]+(input_shape[-1],)
        super(Convolution1D, self).init_param((input_shape[0], 1) + input_shape[1:])

    def output_shape(self, input_shape):
        # input shape: (nb_sample, rows, cols)
        # 2d input shape: (nb_sample, input_depth=1, rows, cols)
        # output shape: (nb_sample, new_rows, new_cols)
        # 2d output shape: (nb_sample, output_depth=new_cols, new_rows, 1)
        input_shape_2D = (input_shape[0], 1) + input_shape[1:]
        output_shape = super(Convolution1D, self).output_shape(input_shape_2D)
        return (output_shape[0], output_shape[2], output_shape[1])

    def _pad_x(self, x, pad):
        # x here is already expanded (i.e. has input_depth channel)
        # Note that we always pad the second dimension with 0
        return self.embed(ZeroPadding2D((pad[0],0)))(x)

    def output(self, x):
        x = self.embed(ExpandDims(axis=0))(x)
        output = super(Convolution1D, self).output(x)[:, :, :, 0]
        output = B.transpose(output, [0, 2, 1])
        return output

    def input_dimension(self):
        return 3


class Convolution2D(ConvolutionBase):
    '''Convolution layer for convolving (input_depth, input_row, input_col) inputs.
    - input_shape: 4D, `(nb_samples, input_depth, input_row, input_col)`
    - output_shape: 4D, `(nb_samples, nb_kernel, output_row, output_col)`
    - parameters:
        - W: `(input_depth, nb_kernel, input_row, input_col)`
        - b: `(nb_kernel,)`

    @note
    `output_row` and `output_col` are determined `pad` and `strides`. For details, please see ConvolutionBase.
    '''
    def __init__(self, nb_kernel, kernel_row, kernel_col, strides=(1, 1), pad='valid', bias=True, init='glorot_uniform', activation='linear', **kwargs):
        '''
        @param nb_kernel: int. Number of convolution kernels to use.
        @param kernel_row: int. The height of the each kernel.
        @param kernel_col: int. The width of the each kernel.
        @param strides: 2D tuple of int. Steps for vertically/horizontally sliding each kernel for convolution.
        @param pad: str, 'valid' of 'same'. See @ref ConvolutionBase.
        @param bias: boolean. Whether to include a bias (i.e. make the layer affine rather than linear).
        @param init: str/function. Function to initialize trainable parameters. See @ref Initializations.md.
        @param activation: str/function. Activation function applied on the output. See @ref Activations.md.
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).
        '''
        self.nb_kernel = nb_kernel
        self.kernel_row = kernel_row
        self.kernel_col = kernel_col
        super(Convolution2D, self).__init__(kernel_shape=(nb_kernel, kernel_row, kernel_col),
                                            strides=strides,
                                            pad=pad,
                                            bias=bias,
                                            init=init,
                                            activation=activation,
                                            **kwargs)

    def input_dimension(self):
        return 4

    def _pad_x(self, x, pad):
        return self.embed(ZeroPadding2D(pad))(x)


class Convolution3D(Layer):
    '''Not implemented yet.
    '''
    # TODO
    pass


class PoolingBase(Layer):
    '''Base layer for pooling layers. Do not use this layer in your code.
    '''
    def __init__(self, mode, pool_size, strides=None, pad='valid', **kwargs):
        '''
        @param mode: str. `'max'` or `'avg'` for max and average pooling respectively.
        @param pool_size: tuple of int. Shape of the pooling window.
        @param strides: tuple of int. Steps for sliding the pooling window.
        @param pad: str, `'valid'` of `'same'`. See descriptions below.
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).

        @note
        The output shape and paddings of pooling ayers is calculated as described in ConvolutionBase.__init__. However, when theano is used as backend, we are unable to conduct tensorflow's asymmetric padding. Current solution is to perform symmetric padding with larger (out of front pad and back pad) padding and truncate the output to match the output shape (determined by the formula).
        '''
        super(PoolingBase, self).__init__(**kwargs)
        assert mode in {'max', 'avg'}, 'Pooling mode must be in {max, avg}'
        assert pad in {'valid', 'same'}, 'pad must be in {valid, same}'

        if strides is None:
            strides = pool_size

        self.mode = mode
        self.pool_size = utils.to_tuple(pool_size)
        self.strides = utils.to_tuple(strides)
        self.pad = pad

    def output_shape(self, input_shape):
        # input shape: (nb_sample, input_depth, rows, cols...)
        # output shape: (nb_sample, input_depth, new_rows, new_cols...)
        output_shape = calc_output_shape(pad=self.pad,
                                         input_shape=input_shape[2:],
                                         kernel_shape=self.pool_size,
                                         strides=self.strides)
        return input_shape[:2] + tuple(output_shape)

    def _calc_pad(self, x):
        if B.name()=='tensorflow':
            return self.pad
        else:
            input_shape = self.get_tensor_shape(x)
            return calc_pad(pad=self.pad,
                            input_shape=input_shape[2:],
                            kernel_shape=self.pool_size,
                            strides=self.strides)

    def output(self, x):
        # Unlike convolutions layers, we do not pad with ZeroPadding layers,
        # but use the padding argument of theano/tensorflow pooling functions.
        # Because user padded values will affect the pooling result.
        pad = self._calc_pad(x)
        return B.pool(x,
                      mode=self.mode,
                      pool_size=self.pool_size,
                      strides=self.strides,
                      padding=pad)


class Pooling1D(PoolingBase):
    '''Pooling layer for sub-sampling (sequence_length, input_dim) inputs.
    - input_shape: 3D, `(nb_samples, sequence_length, input_dim)`
    - output_shape: 3D, `(nb_samples, output_sequence_length, input_dim)`

    @note
    `output_sequence_length` is determined by `pad` and `stride`. For details, please see PoolingBase, ConvolutionBase.
    '''
    def __init__(self, mode, pool_length=2, stride=None, pad='valid', **kwargs):
        '''
        @param mode: str. `'max'` or `'avg'` for max and average pooling respectively.
        @param pool_length: int. The height of the pooling window. The width of the window will always be the input's width.
        @param stride: int. Steps for vertically sliding the pooling window.
        @param pad: str, `'valid'` of `'same'`. See PoolingBase, ConvolutionBase.
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).
        '''
        if stride is None:
            stride = pool_length
        super(Pooling1D, self).__init__(mode=mode,
                                        pool_size=(pool_length, 1),
                                        strides=(stride, 1),
                                        pad=pad,
                                        **kwargs)
        self.pool_length = pool_length
        self.stride = stride

    def output_shape(self, input_shape):
        # input shape: (nb_sample, rows, cols)
        # 2d input shape: (nb_sample, input_depth=1, rows, cols)
        # output shape: (nb_sample, new_rows, cols)
        # 2d output shape: (nb_sample, input_depth=1, new_rows, cols)
        input_shape_2D = (input_shape[0], 1) + input_shape[1:]
        output_shape = super(Pooling1D, self).output_shape(input_shape_2D)
        return (output_shape[0], output_shape[2], output_shape[3])

    def _calc_pad(self, x):
        pad = super(Pooling1D, self)._calc_pad(x)
        if isinstance(pad, tuple):
            assert len(pad)==2
            return (pad[0], 0)
        else:
            return pad

    def output(self, x):
        x = self.embed(ExpandDims(axis=0))(x)
        output = super(Pooling1D, self).output(x)[:, 0, :, :]
        return output

    def input_dimension(self):
        return 3


class Pooling2D(PoolingBase):
    '''Pooling layer for sub-sampling (input_depth, input_row, input_col) inputs.
    - input_shape: 4D, `(nb_samples, input_depth, input_row, input_col)`
    - output_shape: 4D, `(nb_samples, input_depth, output_row, output_col)`

    @note
    `output_row` and `output_col` are determined by `pad` and `strides`. For details, please see PoolingBase, ConvolutionBase.
    '''
    def __init__(self, mode, pool_size=(2, 2), strides=None, pad='valid', **kwargs):
        '''
        @param mode: str. `'max'` or `'avg'` for max and average pooling respectively.
        @param pool_size: tuple of int. Shape of the pooling window.
        @param strides: tuple of int. Steps for sliding the pooling window.
        @param pad: str, `'valid'` of `'same'`. See PoolingBase, ConvolutionBase.
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).
        '''
        super(Pooling2D, self).__init__(mode=mode,
                                        pool_size=pool_size,
                                        strides=strides,
                                        pad=pad,
                                        **kwargs)

    def input_dimension(self):
        return 4


class Pooling3D(PoolingBase):
    '''Zero-padding layer for (input_depth, input_x, input_y, input_z) inputs.
    - input_shape: 5D, `(nb_samples, input_depth, input_x, input_y, input_z)`
    - output_shape: 5D, `(nb_samples, input_depth, output_x, output_y, output_z)`

    @note
    `output_x`, `output_y` and `output_z` are determined by `pad` and `strides`. For details, please see PoolingBase, ConvolutionBase.
    '''
    def __init__(self, mode, pool_size=(2, 2, 2), strides=None, pad='valid', **kwargs):
        '''
        @param mode: str. `'max'` or `'avg'` for max and average pooling respectively.
        @param pool_size: tuple of int. Shape of the pooling window.
        @param strides: tuple of int. Steps for sliding the pooling window.
        @param pad: str, `'valid'` of `'same'`. See PoolingBase, ConvolutionBase.
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).
        '''
        super(Pooling3D, self).__init__(mode=mode,
                                        pool_size=pool_size,
                                        strides=strides,
                                        pad=pad,
                                        **kwargs)

    def input_dimension(self):
        return 5


class ZeroPaddingBase(Layer):
    '''Base layer for zero padding layers. Do not use this layer in your code.
    '''
    def __init__(self, pad_num, axis, **kwargs):
        super(ZeroPaddingBase, self).__init__(**kwargs)

        self.pad_num = pad_num
        self.axis = axis
        all_zero = True
        for p in pad_num:
            if isinstance(p, (tuple, list)):
                all_zero = False
            else:
                all_zero = (all_zero and p==0)
        self.all_zero = all_zero

    def output_shape(self, input_shape):
        # input shape: (nb_sample, input_depth, rows, cols...)
        # pad_shape: symmetric: (k_rows, k_cols...)
        # output shape: (nb_sample, input_depth, new_rows, new_cols...)

        if self.all_zero:
            return input_shape

        output_shape = list(input_shape)
        for i in range(2, len(output_shape)):
            if i not in self.axis:
                continue

            p = self.pad_num[i-2]
            if isinstance(p, (tuple,list)):
                if len(p)!=2:
                    raise KError("You could at most specify 2 padding for insymmetric padding!")
                output_shape[i] += p[0]+p[1]
            else:
                output_shape[i] += 2*p

        return tuple(output_shape)

    def output(self, x):
        if self.all_zero:
            return x

        input_shape = self.get_tensor_shape(x)
        output_shape = ZeroPaddingBase.output_shape(self, input_shape)

        return B.padding(x, self.pad_num, self.axis, output_shape)


class ZeroPadding1D(ZeroPaddingBase):
    '''Zero-padding layer for (sequence_length, input_dim) inputs.
    - input_shape: 3D, `(nb_samples, sequence_length, input_dim)`
    - output_shape: 3D, `(nb_samples, sequence_length+2*pad, input_dim)`
    '''
    def __init__(self, pad, **kwargs):
        '''
        @param pad: int or 2D tuple of int. Number of zeros to pad at the beginning and end of the padding dimension (sequence dimension). If a tuple such as (1,2) is used, we pad 1 zero at the beginning and 2 zeros at the end.
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).
        '''
        super(ZeroPadding1D, self).__init__(pad_num=(pad,), axis=[2], **kwargs)
        self.pad = pad

    def output_shape(self, input_shape):
        # input shape: (nb_sample, rows, cols)
        # 2d input shape: (nb_sample, input_depth=1, rows, cols)
        # output shape: (nb_sample, new_rows, cols)
        # 2d output shape: (nb_sample, output_depth=1, new_rows, cols)
        input_shape_2D = (input_shape[0], 1) + input_shape[1:]
        output_shape = super(ZeroPadding1D, self).output_shape(input_shape_2D)
        return (output_shape[0], output_shape[2], output_shape[3])

    def output(self, x):
        x = self.embed(ExpandDims(axis=0))(x)
        output = super(ZeroPadding1D, self).output(x)[:, 0, :, :]
        return output

    def input_dimension(self):
        return 3


class ZeroPadding2D(ZeroPaddingBase):
    '''Zero-padding layer for (input_depth, input_row, input_col) inputs.
    - input_shape: 4D, `(nb_samples, input_depth, input_row, input_col)`
    - output_shape: 4D, `(nb_samples, input_depth, input_row+2*pad[0], input_col+2*pad[1])`
    '''
    def __init__(self, pad=(1,1), **kwargs):
        '''
        @param pad: tuple. Number of zeros to pad at each dimension. For example (2, (1,3)) will pad 2 zeros at the beginning and the end of the first dimension, and pad 1 zero at the beginning and 3 zeros at the end of the second dimension.
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).
        '''
        super(ZeroPadding2D, self).__init__(pad_num=pad, axis=[2,3], **kwargs)
        self.pad = pad

    def input_dimension(self):
        return 4


class ZeroPadding3D(ZeroPaddingBase):
    '''Zero-padding layer for (input_depth, input_x, input_y, input_z) inputs.
    - input_shape: 5D, `(nb_samples, input_depth, input_x, input_y, input_z)`
    - output_shape: 5D, `(nb_samples, input_depth, input_x+2*pad[0], input_y+2*pad[1], input_z+2*pad[2])`
    '''
    def __init__(self, pad=(1,1,1), **kwargs):
        '''
        @param pad: tuple. Number of zeros to pad at each dimension. For example (2, 2, (1,3)) will pad 2 zeros at the beginning and the end of the first dimension and the second dimension, and pad 1 zero at the beginning and 3 zeros at the end of the third dimension.
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).
        '''
        super(ZeroPadding3D, self).__init__(pad_num=pad, axis=[2,3,4], **kwargs)
        self.pad = pad

    def input_dimension(self):
        return 5


class UnSamplingBase(Layer):
    '''Base layer for unsampling layers. Do not use this layer in your code.
    '''
    def __init__(self, repeats, axis, **kwargs):
        super(UnSamplingBase, self).__init__(**kwargs)
        self.repeats = repeats
        self.axis = axis

    def output_shape(self, input_shape):
        output_shape = list(input_shape[2:])
        for i, r in enumerate(self.repeats):
            output_shape[i]*=r
        return input_shape[:2] + tuple(output_shape)

    def output(self, x):
        for r, a in zip(self.repeats, self.axis):
            x = B.repeat(x, r, a)
        return x


class UnSampling1D(UnSamplingBase):
    '''Repeat each temporal step `length` times along the time axis.
    - input_shape: 3D, `(nb_samples, sequence_length, input_dim)`
    - output_shape: 3D, `(nb_samples, sequence_length*repeat, input_dim)`
    '''

    def __init__(self, repeat=2, **kwargs):
        '''
        @param repeat: int. Factor to repeat.
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).
        '''
        super(UnSampling1D, self).__init__(repeats=[repeat], axis=[2], **kwargs)
        self.repeat = repeat

    def output_shape(self, input_shape):
        # input shape: (nb_sample, rows, cols)
        # 2d input shape: (nb_sample, input_depth=1, rows, cols)
        # output shape: (nb_sample, new_rows, cols)
        # 2d output shape: (nb_sample, output_depth=1, new_rows, cols)
        input_shape_2D = (input_shape[0], 1) + input_shape[1:]
        output_shape = super(UnSampling1D, self).output_shape(input_shape_2D)
        return (output_shape[0], output_shape[2], output_shape[3])

    def output(self, x):
        x = self.embed(ExpandDims(axis=0))(x)
        output = super(UnSampling1D, self).output(x)[:, 0, :, :]
        return output

    def input_dimension(self):
        return 3


class UnSampling2D(UnSamplingBase):
    '''Unsampling layer for (input_depth, input_row, input_col) inputs.
    - input_shape: 4D, `(nb_samples, input_depth, input_row, input_col)`
    - output_shape: 4D, `(nb_samples, input_depth, input_row*repeat[0], input_col*repeat[1])`
    '''
    def __init__(self, repeats=(2,2), **kwargs):
        '''
        @param repeats: tuple of int. Factor to repeat for each dimension.
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).
        '''
        super(UnSampling2D, self).__init__(repeats=repeats, axis=[2,3], **kwargs)

    def input_dimension(self):
        return 4


class UnSampling3D(UnSamplingBase):
    '''Unsampling layer for (input_depth, input_x, input_y, input_z) inputs.
    - input_shape: 5D, `(nb_samples, input_depth, input_x, input_y, input_z)`
    - output_shape: 5D, `(nb_samples, input_depth, input_x*repeat[0], input_y*repeat[1], input_z*repeat[2])`
    '''
    def __init__(self, repeats=(2,2,2), **kwargs):
        '''
        @param repeats: tuple of int. Factor to repeat for each dimension.
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).
        '''
        super(UnSampling3D, self).__init__(repeats=repeats, axis=[2,3,4], **kwargs)

    def input_dimension(self):
        return 5

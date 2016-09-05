[TOC]

# Writing Customized Layer {#writing_customized_layer}
In the following, we describe the basic and advanced issues about writing customized layer.

## Implementing Layer Functions {#implementing_layer_functions}
To make a customized layer works:
1. The layer class should inherit either [Layer](@ref keraflow.layers.base.Layer) (for single input layer) or [MultiInputLayer](@ref keraflow.layers.base.MultiInputLayer) (for multiple input layer).
2. The layer's `__init__` should accept `**kwargs` and call `super(NewLayerClass, self).__init__(**kwargs)` to initialize [common arguments](@ref layer_initialization) of Layer.
3. The following functions should be implemented:

- [init_param(input_shape)](@ref keraflow.layers.base.Layer.init_param): Initialize and register the trainable parameters of the layer. If the layer contains no parameter, you could skip this function. 
- [output(input_tensor)](@ref keraflow.layers.base.Layer.output): Transform input tensor (or a list of input tensors) into an output tensor. Note that the shape of the input tensor could be obtained by [Layer.get_tensor_shape](@ref keraflow.layers.base.Layer.get_tensor_shape). 
- [output_shape(input_shape)](@ref keraflow.layers.base.Layer.output_shape): Return the layer's output shape given the input shape (or a list of input shapes). If not implemented, default behavior is to return the input shape (or the first input shape in the list).
- [input_dimension()](@ref keraflow.layers.base.Layer.input_dimension): For single input layers. Return the expected dimension of the input shape. If not implemented, Keraflow will not check if the input dimension is correct, which could lead to errors at run time.
- [check_input_shape(input_shapes)](@ref keraflow.layers.base.MultiInputLayer.check_input_shape): For multiple-input layers. Validate if the input shapes are correct. If not implemented, default behavior will be checking if all the input shapes are the same. 
- [pack_init_param()](@ref keraflow.layers.base.Layer.pack_init_param): Optional. You should abide some [constraints](@ref serialization_mechanism) on your layer's `__init__` to make it serializable, else you will need to implement this function if you want to correctly serialize the layer.


The following example implements a layer that take an 1D tensor, discard the first half units, and fully connects the second half of the units to the output:
~~~{.py}
from keraflow import utils
from keraflow utils import backend as B

class HalfDense(Layer):
    def __init__(self, output_dim, init='glorot_uniform', activation='linear', **kwargs):
        super(HalfDense, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.init = utils.get_from_module('initializations', init)
        self.activation = utils.get_from_module('activations', activation)

    def input_dimension(self):
        return 2  # Includng batch size dimension.

    def init_param(self, input_shape):
        input_dim = input_shape[1]  # The first dimension is the batch dimension. 
        W = self.init((input_dim/2, self.output_dim))
        self.set_trainable_params('W', W)

    def output(self, x):
        input_shape = self.get_tensor_shape(x)  # Use get_tensor_shape to get the input shape
        input_dim = input_shape[1]  
        output = B.dot(x[:, input_dimension/2:], self.W)  # Dot the second half of the input with W
        return self.activation(output)

    def output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)  # batch size does not change
~~~



## Layer Embedding {#layer_embedding}
Sometimes you might want to use operations implemented by existing layers.
[Layer.embed](@ref keraflow.layers.base.Layer.embed) is a syntax sugar for this purpose.
It embeds the target layer such that the its trainable parameters (along with regularizers and constraints on the parameters) are treated as the host layer's parameters and are updated during the training process.
You could use it in a customized layer's `output()` function like:
~~~{.py}
def output(self, x):
    dx = self.embed(Dense(64, regularizers=['l1']))(x)
~~~

In fact, [SequentialLayer](@ref keraflow.layers.base.SequentialLayer) is implemented using this functions: 
~~~{.py}
class SequentialLayer(Layer):
    def __init__(self, layers=[], **kwargs):
        super(SequentialLayer, self).__init__(**kwargs)
        self.layers = list(layers)

    def output(self, input_tensors):
        output_tensor = input_tensors
        for layer in self.layers:
            output_tensor = self.embed(layer)(output_tensor)
        return output_tensor

    def output_shape(self, input_shapes):
        res = input_shapes
        for layer in self.layers:
            res = layer.output_shape(res)
        return res
~~~
We do not need to implement `init_param` function since `SequentialLayer` itself has no parameters.

You could also check the implementation of the `output` function of [Convolution1D](@ref keraflow.layers.convolution.Convolution1D), [Convolution2D](@ref keraflow.layers.convolution.Convolution2D) and [TimeDistributed](@ref keraflow.layers.wrappers.TimeDistributed) for example on how to use `embed`.


# Keraflow mechanism {#keraflow_mechanism}

## Tensor Linkage Mechanism {#tensor_linkage_mechanism}
The core of tensors linkage is [Layer.__call__](@ref keraflow.layers.base.Layer.__call__).
It accepts a [Kensor](@ref keraflow.layers.base.Kensor) (or as list of Kensors) and return a single [Kensor](@ref keraflow.layers.base.Kensor).
Each kensor embeds a tensor (a theano/tensorflow variable) and the major function of `__call__` is to define the relation between the input tensor(s) and the output tensor.

Another function of `__call__` is to maintain a global list of trainable `parameters`, `regularizers`, `constraints` and parameter `updates`, which will be latter used in [Model.compile](@ref keraflow.models.Model.compile) to determine the total `loss` and what to update during the training process. The task is done by adding current layer's parameters' info into the list carried by the input kensor and then assign the updated list to the output kensor. The model then fetch the list from the output kensor(s).

One final task of `__call__` is to ensure the model could be reconstructed, this is done by keeping a global list of `path` in the form `[path1, path2...]`, where `path1`, `path2`... are in the form of `[input kensor's name, layer's name, output kensor's name]`. When reconstructing the model, we could then find kensors and layers by name and decide to feed which kensor to which layer. The `path` is again passed from kensor to kensor and the model collects the final `path` from the output kensors.


## Serialization Mechanism {#serialization_mechanism}
Keraflow serialize an object (model, layer, regularizer...) by memorizing the object's class, and the value of the arguments listed in its `__init__`.  For un-serialization, we simply call `__init__` of the memorized class with the memorized argument values.

Note that the serialization process is a recursive process, i.e. [serialize](@ref keraflow.utils.generic_utils.serialize)(`obj`) calls [pack_init_params](@ref keraflow.utils.generic_utils.pack_init_param)(`obj`), which in term calls [serialize](@ref keraflow.utils.generic_utils.serialize) on all `__init__` arguments of `obj`.

Due to such implementation, the constraints for an object to be serializable are:
1. The arguments listed in the object's `__init__` should be in the object's `__dict__` when calling [serialize](@ref keraflow.utils.generic_utils.serialize)(`obj`)
2. The arguments listed in the object's `__init__` should be stored __as is__ passed to the layer's `__init__`, i.e. you should not do the following
~~~{.py}
from keraflow.utils import serizlize, unserialize
class MyLayer(Layer):
    def __init__(self, output_dim, additional_dim, **kwargs)
        self.output_dim = output_dim + additional_dim
        self.additional_dim = additional_dim
        ...

layer = MyLayer(1, 2)
layer_config = serialize(d1)
reconstructed = unserialize(layer_config)
~~~
The reason is that when calling [unserialize](@ref keraflow.utils.generic_utils.unserialize)(`layer_config`), `output_dim` is equal to 3 and `additional_dim` equals to 2 . This will make `reconstructed` have `output_dim` equal to 5, which makes `reconstructed` and `layer` behave differently.

If you really need to modify the value of the arguments, you should implement the `pack_init_param` as the following:
~~~{.py}
class MyLayer(Layer):
    def __init__(self, output_dim, additional_dim)
        self.output_dim = output_dim + additional_dim
        self.additional_dim = additional_dim
        ...

    def pack_init_param(self):
        params = super(MyLayer, self).pack_init_param(self)
        params['output_dim'] = params['output_dim']-self.additional_dim
        return params
~~~
Since [serialize](@ref keraflow.utils.generic_utils.serialize) check if an object implements `pack_init_param` before the default behavior of saving the arguments __as is__, the `pack_init_param` of `MyLayer` get called and reset `output_dim` to 1 (as passed during initialization). 

Note that when implementing `pack_init_param` for a new layer, we should first call [Layer.pack_init_param](@ref keraflow.layers.base.Layer.pack_init_param) (line 8), which packs not only arguments listed in `MyLayer.__init__` but also `Layer.__init__` (including regularizers, constraints). Otherwise, you should then handle those things your self. For details, please check the code.

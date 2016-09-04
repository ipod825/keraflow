[TOC]

# An Overview of Layer {#an_over_view_of_layer}

> A Layer is a  one-to-one or many-to-one tensor transformer 

Both theano and tensorflow are graph based deep learning framework.
Let `X`, `W` and `b` be three symbolic tensors.
Then `Y=WX+b` makes `Y` a new symbolic tensor that equals to `W` dots `X` plus `b`.
When there are many tensors in your model, things get messy.
The purpose of [Layer](@ref keraflow.layers.base.Layer) is to simplify the process of building new tensors from existing tensors.
For example:

~~~{.py}
from keraflow.layers import Input, Dense
X = Input(shape=(100,))
dense_layer = Dense(output_dim=64, init='unitorm', activation='linear')
Y = dense_layer(X)  # Y=WX+b
~~~

In Keraflow, an [Input](@ref keraflow.layers.base.Input) layer (line 1) is a special type of layer that could be treated as a symbolic tensor. By feeding a tensor to a layer (line 4), we got a new symbolic tensor Y, which can then be fed to another layer.

## Layer Initialization {#layer_initialization}
Each layer takes different arguments to initialize. As seen above, a dense layer takes three argument. However, there are some common key word arguments you could pass to every layer, which are defined in Layers' [__init__](@ref keraflow.layers.base.Layer.__init__):

~~~{.py}
class Layer(object):
    def __init__(self, 
                 name=None,
                 trainable=True,
                 initial_weights=None,
                 regularizers=None,
                 constraints=None):
~~~

1. Set `name` for easy debugging.
2. Set `trainable` to False, if you don't want the layer parameters be updated during training process.
3. Set `initial_weights` to directly assign the initial values of the layer parameters (this will override the `init` argument). See @ref Initializations.md.
4. Set `regularizers` to apply regularization penalty on the layer parameters. See @ref Regularizers.md.
5. Set `constraints` to restrict the value of layer parameters when updating them during training process. See @ref Constraints.md.

For details on setting `initial_weights`, `regularizers`, and `constraints`, please see @ref argument_passing_summarization and the related pages. 

## Shared Layer {#shared_layer}
A layer can be fed multiple times by different input tensors. Considering the following case:
~~~{.py}
# symbolic tensor
Y1=WA+b
Y2=WB+b

# Keraflow equivalent
# X1, X2 should be of the same shape
A = Input(100)
B = Input(100)
d = Dense(50)
Y1 = d(A)
Y2 = d(B)
~~~

By feeding the same layer two times, we keep only one `W` and `b` in our tensor graph.

## Multiple Inputs to a Layer {#mulriple_inputs_to_a_layer}
A layer might take more than one tensor as input. For example:

~~~{.py}
from keraflow.layers import Input, Dense, Concatenate
Y1 = Dense(50)(Input(100))
Y2 = Dense(20)(Input(100))
concat = Concatenate(axis=0)([Y1, Y2)])  # concat will be a tensor of dimension 70
output = Dense(2)(concat)
~~~

The only difference of a multiple input layer is that it takes a list of input tensors as input instead a single tensor.
Users could define their own layers to take either single or multiple input tensors. See @ref implementing_layer_functions


## Anonymous Layers (Sequential) {#anonymous_layers}
We already see how to obtain a new tensor by feeding a tensor to a layer. However, it would be cumbersome to name all the tensors if we just want to perform a series of operation. [Sequential](@ref keraflow.models.Sequential) thus provides a syntax sugar for us:

~~~{.py} 
from keraflow.layers import Input, Dense, Activation
from keraflow.models import Sequential

seq = Sequential()
seq.add(Input(100))  
seq.add(Dense(64))
seq.add(Dense(32))
seq.add(Activation('softmax'))

# Alternatively
seq = Sequential([Input(100), Dense(64), Dense(32), Activation('softmax')])
~~~

Note that when the first layer of `Sequential` is an `Input` layer, it is treated as a tensor (the tensor output by the last layer), i.e. you could feed it to another layer but you can not feed other tensor to it. 
~~~{.py} 
from keraflow.layers import Input, Dense, Activation
from keraflow.models import Sequential

seq = Sequential([Input(100), Dense(64), Dense(32), Activation('softmax')])
output = Dense(2)(seq)
~~~

When the first layer of `Sequential` is not an `Input` layer, it is treated as a normal layer, whether it takes a single tensor or a list of tensors denpend on its first layer.

~~~{.py}
seq = Sequential()
seq.add(Concatenate())  
seq.add(Dense(64))

input1 = Input(100)
input2 = Input(100)

output = seq([input1, input2])
~~~

We've cover what users need to know to use existing layers in Keraflow. For advanced information such as writing customized layer, please refer to @ref Developer-Guide.md


# An Overview of Model {#an_over_view_of_model}
A typical deep learning model does the follows:

1. Derives the output tensor `Y` from the input tensor `X` such that `Y` = f(`X`), where f is the model with some trainable parameters (e.g. `Y`=`WX`+`b`).
2. Decides how far `Y` is from the gold answer `Target` according to a loss function: `loss` = L(`Y`, `Target`). 
3. Calculates the gradient of `loss` with respect to the trainable parameters: `Gw`, `Gb` = grad(`loss`, `W`), grad(`loss`, `b`)
4. Minimizes `loss` according to some optimizing rule (e.g. subtracting the gradient from the parameters `W`, `b` = `W`-`GW`, `b`-`Gb`)

We now introduce [Model](@ref keraflow.models.Model) and [Sequential](@ref keraflow.models.Sequential) to cover these steps.

## Constructing the Model {#constructing_the_model}
Let's start from the simple model with one input tensor:
~~~{.py}
from keraflow.layers import Input, Dense
from keraflow.models import Model

X = Input(100)
M = Dense(50)(X)
Y = Dense(2, activation='softmax')(X)
model = Model(inputs=X, outputs=Y)  

~~~

We tell the model that the input tensor is `X` and the output tensor is `Y`. In the one-input-one-output case, we could simply use [Sequential](@ref keraflow.models.Sequential) to avoid naming all the tensors.

~~~{.py}
from keraflow.layers import Input, Dense
from keraflow.models import Sequential
model = Sequential()
model.add(Input(100))
model.add(Dense(50))
model.add(Dense(2, activaion='softmax'))

# Alternatively 
model = Sequential([Input(100), Dense(50), Dense(2, activation='softmax')])
~~~

Note that [Sequential](@ref keraflow.models.Sequential) can only be used as a model (able to call `fit`, `predict`) when its first layer is an [Input](@ref keraflow.layers.base.Input) layer. Moreover, if there are multiple input tensors or multiple output tensors, we could only use `Model`:

~~~{.py}
from keraflow.layers import Input, Dense
from keraflow.models import Sequential, Model

X1 = Input(100)
Y1 = Dense(50, name='output1')(X)
X2 = Input(100)
Y2 = Dense(50, name='output2')(X)
model = Model(inputs=[X1, X2], outputs=[Y1, Y2])  
~~~

## Configuring Loss and Optimizing Rule{#configuring_loss_and_optimizing_rule}
Now we can specify the loss function and the optimizer. For single output model (including [Sequential](@ref keraflow.models.Sequential)):
~~~{.py}
model.compile(optimizer='sgd', loss='categorical_crossentropy')
~~~

For a model with multiple outputs, we need to specify a loss function for each output channel:
~~~{.py}
model.compile(optimizer='sgd', loss={'output1':'categorical_crossentropy', 'output2': 'mean_squared_error'})
~~~
Note that the name of each output channel is the name the corresponding output layer. If you feel that writing these names unnecessary, you could also pass a list to `loss` (see @ref Objectives.md).

In the examples, we pass strings (e.g. `sgd`, `categorical_crossentropy`). These strings are actually alias of predefined optimizers and loss functions. You could also pass customized optimize/loss function instance (see @ref Optimizers.md, @ref Objectives.md).


## Train/Evaluate/Predict {#train_evaluate_predict}
Both [Model](keraflow.models.Model) and [Sequential](keraflow.models.Sequential) has 
- [fit](@ref keraflow.models.Model.fit): fitting `X` to `Target` by iteratively updating model parameters.
- [evaluate](@ref keraflow.models.Model.evaluate): output `loss` given `X` and `Target`
- [predict](@ref keraflow.models.Model.predict): output `Y` given `X`

Both `fit` and `evaluate` takes `X` and `Target` as inputs. As for `predict`, only `X` is required.

For `Sequential` (single input, single output), `X`, `Target` takes a single numpy array (or list). For `Model`, `X`, `Target` takes a dictionary/list of numpy array(s) (see @ref argument_passing_summarization).

## Model Serialization {#model_serialization}
You could save a (trained) model to disk and restore it for later use. Simply run
~~~{.py}
model.save_to_file(arch_fname='arch.json', weight_fname='weight.pkl', indent=2)
~~~

Note that the architecture and the model parameters are stored separately.  If you don't want to save model weights, set `weight_fname` to `None`.

Keraflow supports two file formats for storing model architectue: `json` and `yaml`. Please specify the file extension to switch between these two formats.
Note that in addition to `arch_fname` and `weight_fname`, `save_to_file` takes `**kwargs` and pass them to `json.dump` or `yaml.dump`. So, in the above example, the `indent` argument is to beautify the architecture output. 



# Argument Passing Summarization {#argument_passing_summarization}
To bring both flexibly and convince to argument passing in Keraflow, for some arguments of some functions, users can pass a single value, a dictionary, or a list. We summarize them as follows. For more examples, please refer to [Initial Weights](@ref initialization-of-layer-parameters-by-weights),  @ref Regularizers.md, and @ref Constraints.md.

##  Array-like data:
Related arguments:
1. [Layer.__init__](@ref keraflow.layers.base.Layer.__init__): `initial_weights`
2. [fit](@ref keraflow.models.Model.fit), [predict](@ref keraflow.models.Model.predict), [evaluate](@ref keraflow.models.Model.evaluate): `x`, `y`, `sample_weights`

~~~{.py}
# dict: accepts both array and list
initial_weights = {'W':numpy.array([[1,2],[3,4]]), 'b': numpy.array([1,2])}
initial_weights = {'W':[[1,2],[3,4]], 'b': [1,2]}

# list: accepts only array
initial_weights = [numpy.array([[1,2],[3,4]]), numpy.array([1,2])]

# single value: accepts both array or list 
initial_weights = numpy.array([[1,2], [3,4]])
initial_weights = [[1,2], [3,4]]
~~~

## Function or str or Class instance
Related arguments:
1. [Layer.__init__](@ref keraflow.layers.base.Layer.__init__): `regularizers`, `constraints`
2. [compile](@ref keraflow.models.Model.compile): `loss`, `metrics`

~~~{.py}
# dict: accepts functions/str/class instance
metrics = {'output1': metrics.accuracy}
loss = {'output1': 'mse', 'output2':'mae'}
regularizers = {'W': regularizers.L1(0.5), 'b': regularizers.L2(0.1)}

# list: accepts functions/str/class instance
metrics = [metrics.accuracy]
loss = ['mse', 'mae']
regularizers = [regularizers.L1(0.5), regularizers.L2(0.1)]

# single value: Only loss accepts single value, if there is only a single output.
loss = 'categorical_crossentropy'
~~~


## Rules for Missing values:
Related arguments:
1. [Layer.__init__](@ref keraflow.layers.base.Layer.__init__): `initial_weights`, `regularizers`, `constraints`
2. [fit](@ref keraflow.models.Model.fit), [predict](@ref keraflow.models.Model.predict), [evaluate](@ref keraflow.models.Model.evaluate): `sample_weights`
3. [compile](@ref keraflow.models.Model.compile): `metrics`

~~~{.py}
# dict: specify only the used key
initial_weights = {'W':[[1,2],[3,4]]}

# list: first  come first serves
initial_weights = [numpy.array([[1,2],[3,4]])] # sets only W's weights

# single value: first come first serves 
# useful when there is only one parameter/input layer/output layer
initial_weights = [[1,2],[3,4]] # sets only W's weights
~~~

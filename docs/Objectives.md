[TOC]

# Usage of Objectives {#usage_of_objectives}
An objective function is used to measure how bad (or good) the model is. Concretely, objective functions measures how "far" (or "close") between the model's output and the gold answer. Based on the objective function, the model adjusts its parameters to minimize (or maximize) the score returned by the objective function. Usually, we use the name "loss function" when it is to be minimized.

Model's [fit function](@ref keraflow.models.Model.fit) requires one (or more, for multi-output models) loss function for the `loss` argument so that the model can minimize it. Additionally, you could monitor one additional objective function value for each output channel by the `metric` argument. Concretely:

~~~{.py}
input1 = Input(1, name='input1')
input2 = Input(1, name='input2')
output1 = Dense(1, name='output1')(input1)
output2 = Dense(1, name='output2')(input2)

model = Model([input1, input2], [output1,output2])
model.compile('sgd', loss={'output1':'mse', 'output2':'mse'}, metrics={'output1':'acc'})
~~~

Note that you must provide one objective function for each output channel, while you could skip some (or all) output channels for `metric`. Also note that the key to refer an output channel is bound to the name of the outputting layer of that channel.

Note that `mse` stands for `squared_error` and `acc` stands for `acuracy`. Keraflow uses alias for some build-in objective functions (see below). You could either use the full name or the alias name. Alternatively, You can also pass a Theano/TensorFlow function to the `loss` argument:

~~~{.py}
from keraflow import backend as B

def absolute_error(y_pred, y_true):
    return B.abs(y_pred - y_true)

model.compile('sgd', loss={'output1':'mse', 'output2':absolute_error}, metrics={'output1':'acc', 'output2':absolute_error})
~~~

Note that the function must take two arguments: 
1. `y_true`: True labels. 2D (or more) Theano/TensorFlow tensor
2. `y_pred`: Theano/TensorFlow tensor of the same shape as `y_true`.

The first dimension of the input tensor is the sample dimension. The function should not average over samples since it is done by Keraflow later. Therefore, the function must return an 2D (or more) tensor with the first dimension the sample dimension.

In addition to dict, you could also pass a list to `loss` & `metric`:

~~~{.py}
model = Sequential([Input(1), Dense(1)])
model.compile('sgd', loss=['mse', absolute_error], metrics=['acc'])
~~~

You could only specify metrics for partial output channels. For the dict method, just specify the target output names. For the list method, we use the first come first serve strategy, i.e. `["acc"]` will only specify accuracy metric for `output1`. 


# Available Objectives {#available_objectives}
- [square_error/se/mse](@ref keraflow.objectives.square_error)
- [absolute_error/ae/mae](@ref keraflow.objectives.absolute_error)
- [absolute_percentage_error/ape/mape](@ref keraflow.objectives.absolute_percentage_error)
- [squared_logarithmic_error/sle/msle](@ref keraflow.objectives.squared_logarithmic_error)
- [hinge](@ref keraflow.objectives.hinge)
- [squared_hinge](@ref keraflow.objectives.squared_hinge)
- [binary_crossentropy](@ref keraflow.objectives.binary_crossentropy)
- [categorical_crossentropy](@ref keraflow.objectives.categorical_crossentropy)
- [kullback_leibler_divergence](@ref keraflow.objectives.kullback_leibler_divergence)
- [poisson](@ref keraflow.objectives.poisson)
- [cosine_proximity](@ref keraflow.objectives.cosine_proximity)

@note
For the alias of `m*`, `m` stands for mean (over sample points). As said, Keraflow objective function does not average over sample points, so the name of the function does not contain `mean`. However, to follow machine learning convention, we still provide alias starting with `m`.

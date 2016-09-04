[TOC]

# Usage of Activations {#usage_of_activations}
Activation function adds non-linearity to each layer's output. To specify the activation function of a layer. You could either appends after the layer an [Activation layer](@ref keraflow.layers.core.Activation)

~~~{.py}
model = Sequential([Input(64), Dense(2), Activation('relu')])
~~~

or through the `activation` argument of the layer:

~~~{.py}
model = Sequential([Input(64), Dense(2, activation='relu')])
~~~

You can also pass a Theano/TensorFlow function to the `actication` argument:

~~~{.py}
from keraflow import backend as B

def truncated_relu(x):
    return B.relu(x, max_value=20)

model = Sequential([Input(64), Dense(2, activation=truncated_relu)])
~~~

or utilize an existing Keraflow activation function:
~~~{.py}
from keraflow import backend as B
from keraflow.activations import relu

def truncated_relu(x):
    return relu(x, max_value=20)

model = Sequential([Input(64), Dense(2, activation=truncated_relu)])
~~~

# Available Activations {#available_activations}

- [linear](@ref keraflow.activations.linear)
- [sigmoid](@ref keraflow.activations.sigmoid)
- [hard_sigmoid](@ref keraflow.activations.hard_sigmoid)
- [tanh](@ref keraflow.activations.tanh)
- [relu](@ref keraflow.activations.relu)
- [softmax](@ref keraflow.activations.softmax)
- [softplus](@ref keraflow.activations.softplus)
- [softsign](@ref keraflow.activations.softsign)


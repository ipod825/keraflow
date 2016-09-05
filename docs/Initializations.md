[TOC]

# Initialization of Layer Parameters by Weights {#initialization-of-layer-parameters-by-weights}

You could initialize a layer's parameters with numpy arrays by setting the layer's `initial_weights` argument 
~~~{.py}
input = Input(2)
dense = Dense(2, initial_weights={'W':np.array([[1,2],[3,4]]), 'b':np.array([1,2])})
dense_output = dense(input)
~~~

Note that you could also pass a list to the `initial_weights` argument.

~~~{.py}
dense = Dense(2, initial_weights=[np.array([[1,2],[3,4]]), np.array([1,2])})
~~~

To know the number of parameters, their name, and their order (for applying the list initialization method), you need to check the manual for each layer.

You could also specify `initial_weights` for partial parameters. For the dict method, just specify the parameters to be adopted. For the list method, we use the first come first serve strategy, i.e. `[np.array([[1,2],[3,4]])]` will only specify weights for `W`.

Note that, for flexibility, `initial_weights` accepts single nD list or a single numpy array.

~~~{.py}
dense = Dense(2, initial_weights=[[1,2],[3,4]])
dense = Dense(2, initial_weights=np.array([[1,2],[3,4]]))
~~~

Both are equivalent to:

~~~{.py}
dense = Dense(2, initial_weights=[np.array([[1,2],[3,4]])])
~~~

However, this only specifies initial weight for the first parameter and hence  should be use with care. 



# Initialization of Layer Parameters by Function {#initialization-of-layer-parameters-by-function}
You could also initialize layer parameters with initializing function. Keraflow implements some 'rule-of-thumb' methods as listed below. You could set the initializing function thorough each layer's `init` argument.


~~~{.py}
model = Sequential([Input(64), Dense(2, init='uniform')])
~~~

You can also pass an Theano/TensorFlow function to the `init` argument:

~~~{.py}
from keras import backend as B
import numpy as np

def my_uniform(shape, name=None):
    scale = 0.1
    return B.variable(np.random.uniform(low=-scale, high=scale, size=shape), name=name)

model = Sequential([Input(64), Dense(2, init=my_init)])
~~~

or utilize an existing Keraflow initializing function:
~~~{.py}
from keraflow import backend as B
from keraflow.initializations import uniform

def my_uniform(shape, name):
    return uniform(scale=0.1, name=name)

model = Sequential([Input(64), Dense(2, init=my_uniform)])
~~~

Note that the function must take two arguments and return a Theano/Tensorflow variable: 
1. `shape`: shape of the variable to initialize
2. `name`: name of the variable


# Available Initializations {#available-initializations}

- [uniform](@ref keraflow.initializations.uniform)
- [lecun_uniform](@ref keraflow.initializations.lecun_uniform) 
- [normal](@ref keraflow.initializations.normal)
- [identity](@ref keraflow.initializations.identity)
- [orthogonal](@ref keraflow.initializations.orthogonal)
- [zero](@ref keraflow.initializations.zero)
- [one](@ref keraflow.initializations.one)
- [glorot_normal](@ref keraflow.initializations.glorot_normal)
- [glorot_uniform](@ref keraflow.initializations.glorot_uniform)
- [he_normal](@ref keraflow.initializations.he_normal)
- [he_uniform](@ref keraflow.initializations.he_uniform)

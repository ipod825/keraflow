[TOC]

# Usage of Regularizers {#usage_of_regularizers}
Unlike [loss functions](@ref Objectives.md) which measure the model performance, regularizer functions, measure how "bad" (in the sense of possibility of overfitting) the model parameters are and apply penalties on the parameters. These penalties are then incorporated in the objective functions and the network simultaneously minimized the loss and the penalties.


To _regularize_ a layer's parameter, you could specify the `regularizers` argument of the layer. 

~~~{.py}
model = Sequential([Input(64), Dense(2, regularizers={'W':'l1', 'b':'l2'})])
~~~
Note that you could also pass a list to the `regularizers` argument.

~~~{.py}
model = Sequential([Input(64), Dense(2, regularizers=['l1', 'l2'])])
~~~

To know the number of parameters, their name, and their order (for applying the list initialization method), you need to check the manual for each layer.

You could also specify regularizers for partial parameters. For the dict method, just specify the parameters to be adopted. For the list method, we use the first come first serve strategy, i.e. `["l1"]` will only specify L1 regularizer for `W`. 


In above examples, `l1` stands for `L1`. Keraflow use alias for some build-in regulzrizers (see below). You could either use the full name or the alias name. Alternatively, you can also pass a regularizer instance to the `regularizers` argument:

~~~{.py}
from keraflow.regularizers import L1

model = Sequential([Input(64), Dense(2, regularizer=[L1(0.05)])])
~~~


# Available regularizers {#available_regularizers}
- [L1/l1](@ref keraflow.regularizers.L1)
- [L2/l2](@ref keraflow.regularizers.L2)
- [L1L2/l1l2](@ref keraflow.regularizers.L1L2)

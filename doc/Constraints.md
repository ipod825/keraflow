[TOC]

# Usage of Constraints {#usage_of_constraints}
During back propagation, there could be "gradient-explosin" problem which makes neural network hard to converge. A method to deal with the problem is to add constraint to the parameter updating process (e.g. constrain the maxnorm of the updated parameter).

To _constrain_ a layer's parameter, you could specify the `constraints` argument of the layer. 

~~~{.py}
model = Sequential([Input(64), Dense(2, constraints={'W':'maxnorm', 'b':'nonneg'})])
~~~

Note that you could also pass a list to the `constraints` argument.

~~~{.py}
model = Sequential([Input(64), Dense(2, constraints=['maxnorm', 'nonneg'])])
~~~

To know the number of parameters, their name, and their order (for applying the list initialization method), you need to check the manual for each layer.

You could also specify constraints for partial parameters. For the dict method, just specify the parameters to be adopted. For the list method, we use the first come first serve strategy, i.e. `["maxnorm"]` will specify `MaxNorm` constraint for `W` and leave `b` plain. 

In above examples, `maxnorm` stands for `MaxNorm`. Keraflow use alias for some build-in constraints (see below). You could either use the full name or the alias name. Alternatively, you can also pass a constraint instance to the `constraints` argument:

~~~{.py}
from keraflow.constraints import MaxNorm

model = Sequential([Input(64), Dense(2, constraint=[MaxNorm(m=3, axis=1)])])
~~~


# Available Constraints {#available_constraints}
- [MaxNorm/maxnorm](@ref keraflow.constraints.MaxNorm)
- [NonNeg/nonneg](@ref keraflow.constraints.NonNeg)
- [UnitNorm/unitnorm](@ref keraflow.constraints.UnitNorm)

[TOC]

# Usage of Optimizers {#usage_of_optimizers}
An optimizer tries to minimize score returned by loss function adjusting model parameters.
Keraflow model requires user to specify the optimizers to configure the model.
~~~{.py}
model.compile(loss='mse', optimizer='sgd')
~~~

Note that `sgd` stands for `SGD`. Keraflow uses alias for some build-in optimizers (see below). You could either use the full name or the alias name. Alternatively, You can also pass an optimizer instance to the `optimizer` argument:

~~~{.py}
from keraflow.optimizers import SGD

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd)
~~~


# Available Optimizers{#available_optimizers}
- [SGD/sgd](@ref keraflow.optimizers.SGD)
- [RMSprop/rmsprop](@ref keraflow.optimizers.RMSprop)
- [Adagrad/adagrad](@ref keraflow.optimizers.Adagrad)
- [Adadelta/adadelta](@ref keraflow.optimizers.Adadelta)
- [Adam/adam](@ref keraflow.optimizers.Adam)
- [Adamax/adamax](@ref keraflow.optimizers.Adamax)

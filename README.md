# Keraflow: Deep Learning library for Theano and Tensorflow. Keras follower.

## Why Keraflow

> Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either [TensorFlow](https://github.com/tensorflow/tensorflow) or [Theano](https://github.com/Theano/Theano). It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

As its official description, [Keras](https://github.com/fchollet/keras) serves as an excellent front-end for graph based deep learning framework. However, from the point of view of soft-engineering, its API design and its complicated internal mechanism makes it hard to understand and cumbersome to extend. Therefore, I reimplement the core of Keras to provide:

1. An simpler tensor linkage mechanism for developers to understand it.
2. A cleaner and more consistent API for user to use and extend it.
3. Some extra functionality not implemented in Keras.

For the full API reference, read the [online documentation](https://ipod825.github.io/keraflow/docs/html/index.html).
It is strongly recommended to read the [Tutorials](https://ipod825.github.io/keraflow/docs/html/md_Tutorials.html) first to know the basics on building neural network models with Keraflow.

---
## Features
### Keras-like
- Uses both theano and tensorflow as backend.
- Provides various layers including convolution layers and recurrent layers.
- Decoupled regularizer, constraint, layer, optimizer... modules.
- Supports arbitrary connectivity schemes (including multi-input and multi-output training).

### New Features
- A simpler tensor linkage mechnism.
- An easier way of writing custmoized layers
    - No `initial_weights`, `regularizers`, `constraints` arguments for layers' `__init__`.
    - No  `get_config` for layers' serialization.
    - Existing layer are reusable to build new layers with the magic __embed__ function.

For more details about the difference between Keraflow and Keras, please refer to the [Differences from Keras](#differences-from-keras).

---

## Installation

Keraflow uses the following dependencies:

- numpy, scipy, tqdm
- pyyaml, hickle (optional, required if you use model saving/loading functions)
- Optional but recommended if you use CNNs: cuDNN.

*When using the Theano backend:*

- Theano
    - [See installation instructions](http://deeplearning.net/software/theano/install.html#install).

*When using the TensorFlow backend:*

- TensorFlow
    - [See installation instructions](https://github.com/tensorflow/tensorflow#download-and-setup).

To install Keraflow, `cd` to the Keraflow folder and run the install command:
~~~{.bash}
$ sudo python setup.py install
~~~

You can also install Keraflow from PyPI:
~~~{.bash}
$ sudo pip install keraflow
~~~

---
## Contributing

### Environment Setting
To make Keraflow compatible with both python2 & python3. We use [pyenv](https://github.com/yyuu/pyenv) to build virtual environment.
The shell script `dev_scrips/install_dependency.sh` could quickly sets up the testing environments.
~~~{.bash}
# In project root directory (parent directory  of `dev_scripts`)
$ bash dev_scrips/install_dependency.sh
# Two pyenv environment k2, k3 are installed now.
~~~

__Note__: The script does not add pyenv PATH in your script config file (e.g. ~/.zshrc). You will need to manually copy and paste the following into your shell config file so that the next time you log in, pyenv will be in the PATH:

~~~{.bash}
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
~~~

To quickly modify and run tests. Run:
~~~{.bash}
$ python setup.py develop  # will link the project root into k2's site package directory
~~~

And then run tests via `dev_scripts/run_test.sh`. Run:
~~~{.bash}
# In project root directory (parent directory  of `dev_scripts`)
$ bash dev_scripts/run_test.sh
~~~

`dev_scripts/run_test.sh` checks pep8, python2 testing and python3 testing. You could also run tests manually:
~~~{.bash}
$ pyenv activate k2 # python2 environment for testing keraflow
$ py.test tests  # run tests in tests/ directory
$ pyenv activate k3 # python3 environment for testing keraflow
$ py.test tests  # run tests in tests/ directory
~~~


### PEP8
`dev_scripts/run_test.sh` checks pep8, it you fail on syntax check, you could use `autopep8` to fix it:
~~~{.bash}
$ pip install autopep8
$ autopep8 --recursive -i --select E128 keraflow  # fix all error no.128
~~~

It is highly recommend you avoid these errors when writing them using some editor plugins. If you use vim (or neovim), I recommend installing `flake8` and adopt the settings in this [gist](https://gist.github.com/ipod825/fbee70d8bd063f228951cd4b6f38f4df). Note that `flask8` is required:
~~~{.bash}
$ pip install flask8
~~~


### Documentation
1. Documentation uses [doxygen](https://www.stack.nl/~dimitri/doxygen/manual/docblocks.html).
2. Follow the convention to use `@` for special commands (`param`, `package`... etc.)
3. Installation
~~~{.bash}
$ sudo apt-get install doxygen
$ sudo pip install doxypy
~~~
4. Documentation generation
~~~{.bash}
$ cd doc
$ doxygen Doxyfile
# open ./html/index.html with web browser
~~~

---

## Differences from Keras

### A simpler tensor linkage mechanism
Two main things that makes Keras complicated:

1. Keras determines layer input shape by back-tracing from output tensors to input tensors, which makes its tensor linkage process a recursive process.
2. Keras keeps track of tensor linkage by keeping two lists `inbound_nodes` and `outbound_nodes` in each layer. However, maintaining a list to keep track of linkage is usually a brain-killing job.

Keraflow, intead

1.  Flows information (tensors and their shapes) from inputs to outputs, making tensor linkage a sequential process (well, at least `a+b+c+d` is more amiable than `(a+(b+(c+d)))`).
2. Uses only a list to keep track of tensor linkage, which is actually unnecessary if serilization is not a concern.

### An easier way of writing custmoized layers

#### Less arguments for layers' `__init__`
Check the constructor of __Dense__ layer:

__Keras__
~~~{.py}
def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, **kwargs):
~~~

__Keraflow__
~~~{.py}
def __init__(self, output_dim, init='glorot_uniform', activation='linear', bias=True, **kwargs):
~~~

The signal of `initial weights`, `regularizers`, and `constraints` disappear since Keraflow takes care of them in __Layer__ class.
The signal of input dimension also disappears since Keraflow force users to specify an __Input__ layer and their shape for all models.

When creating a customized layer, users no longer need to write `regularizers`, `constraints` initialization code for the layer. Special care for the input dimension is also unnecessary.

One additional merit of abstracting `initial_weights`, `regularizers`, `constraints` initialization process in __Layer__ class is that Keraflow easily (without adding too much code) enables users to initialize those of a layer with dictionary:

~~~{.py}
dense = Dense(64, initial_weights={'W': W, 'b':b},
              regularizers={'W': 'l1', 'b':'l2'},
              constraints={'W': 'maxnorm', 'b':'unitnorm'})
dense = Dense(64, initial_weights=[W, b],
              regularizers=['l1', 'l2'],
              constraints=['maxnorm', 'unitnorm'])
~~~

#### No `get_config` for serialization.
Every layer in Keras has a `get_config` function, which is needed for serializing models. Though its implementation is not necessary for customized layers, it would be good for developers to save the time implementing it just for serializing their models.

Keraflow takes care of this, every layer that fulfils some [constraints](https://ipod825.github.io/keraflow/docs/html/md_Developer-Guide.html#serialization_mechanism) is naturally seizable.

#### Embed existing layers to write new layers
Currently, in Keras, when writing you own layers, even if you want to conduct similar operation of the `Dense` layer, you still need to define some trainable parameters (write initialization code and add it to the layer's trainable parameters list) for that.

In Keraflow, you could simply write (in `output` function, the correspondence of `get_output_for` in Keras):
~~~{.py}
self.embed(Dense(output_dim))(input_tensor)
~~~

Everything is done!! The parameters of `Dense` is automatically added as parameters of your layer and is updated during training. For more information, see [Layer Embedding](https://ipod825.github.io/keraflow/docs/html/md_Developer-Guide.html#layer_embedding)

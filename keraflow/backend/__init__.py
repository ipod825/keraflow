import inspect
import os
import sys

from .common import _EPSILON, _FLOATX
from .. import konfig


def _get_backend_name():
    config = konfig.get_config()
    backend = config.get('backend', 'theano')
    epsilon = config.get('epsilon', _EPSILON)
    floatx = config.get('floatx', _FLOATX)

    assert floatx in {'float16', 'float32', 'float64'}
    assert type(epsilon) == float
    assert backend in {'theano', 'tensorflow'}

    config = {'floatx': floatx,
              'epsilon': epsilon,
              'backend': backend}

    konfig.save_config(config)
    if 'KERAFLOW_BACKEND' in os.environ:
        assert backend in {'theano', 'tensorflow'}
        backend = os.environ['KERAFLOW_BACKEND']

    return backend, config


def _set_backend():
    name, config = _get_backend_name()
    config['name'] = name
    if name == 'theano':
        sys.stderr.write('Using Theano backend.\n')
        from .theano_backend import TheanoBackend as _B
    elif name == 'tensorflow':
        sys.stderr.write('Using Tensorflow backend.\n')
        from .tensorflow_backend import TensorflowBackend as _B
    else:
        assert False, "Invalid backend name: {}".format(name)

    _backend = _B(**config)

    for name, f in inspect.getmembers(_backend):
        if not name.startswith('_'):
            globals()[name] = f

_set_backend()

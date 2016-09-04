'''
@package keraflow.utils.generic_utils
Generic utils for keraflow.
'''
import six
import sys
import os

from . import alias
from .exceptions import KeraFlowError as KError


def to_list(x):
    if not isinstance(x, list):
        return [x]
    else:
        return x


def to_tuple(x):
    if not isinstance(x, tuple):
        if isinstance(x, list):
            return tuple(x)
        else:
            return (x,)
    else:
        return x


def unlist_if_one(x):
    assert isinstance(x, list)
    if len(x)==1:
        return x[0]
    else:
        return x


def concat_lists(lists):
    assert isinstance(lists, list), "lists should be a list of list"
    res_list = []
    for l in lists:
        res_list += l
    return res_list


def dict_unify(dicts):
    '''Merge dicts so that the resulting dict contains all the keys of the input dicts.
    @param dicts: list of dicts.
    '''
    res_dict = {}
    for d in dicts:
        for key, value in d.items():
            if key in res_dict and res_dict[key]!=value:
                raise KError("The merged dictionary has confliction. Key:{}. Values:{}, {}".format(key, value, res_dict[key]))
            else:
                res_dict[key] = value
    return res_dict


def average_dicts(dicts):
    '''Average the values for each key in the dicts.
    @param dicts: list of dicts.
    '''
    assert len(dicts)>0, "Input no dicts!!"
    res_dict = dicts[0].copy()
    for d in dicts[1:]:
        for key, value in d.items():
            if key not in res_dict:
                raise KError("Input dicts should have same keys. The key '{}' is not.".format(key))
            else:
                res_dict[key] += value
    for key in res_dict:
        res_dict[key]/=len(dicts)
    return res_dict


def get_from_module(module_name, identifier, call_constructor=True, **kwargs):
    '''Return class (or instance of the class) from keraflow's module with identifier (str).
    @param module_name: str. The module name where the target class is located. If a keraflow module, `keraflow.` can be omitted, e.g. `keraflow.activations` could simply be `activations`.
    @param identifier: str/object. The name of the target class. If an object is passed, we simply returned it.
    @param call_constructor: boolean. If True, call the target class' constructor with `kwargs`
    @param kwargs: keyword arguments. Arguments for instantiating the class.
    '''
    import importlib
    try:
        module = importlib.import_module('keraflow.'+module_name)
    except ImportError:
        module = importlib.import_module(module_name)

    res = identifier
    if isinstance(res, six.text_type) or isinstance(res, str):
        res = alias.get_original(module_name, res)
        if res not in module.__dict__:
            raise Exception("Invalid identifier '{}' for {}".format(identifier, module_name))
        else:
            res = module.__dict__[res]

    if type(res) == type:
        if call_constructor:
            return res(**kwargs)
        else:
            return res
    else:
        return res


def pack_init_param(obj, include_kwargs=[]):
    '''Keraflow's default `__init__` packer.
    Pack the values of arguments of obj's `__init__` by calling `serialize` on each argument.
    @param obj: object to be packed.
    @param include_kwargs: list of str. Names of keyword arguments to be packed. Useful for packing parent class's arguments.
    '''
    import inspect

    cls = obj.__class__
    if '__init__' in cls.__dict__:
        arg_list = inspect.getargspec(cls.__init__)[0]
        arg_list.remove('self')
    else:
        arg_list = []

    init_config = {}
    for name in arg_list+include_kwargs:
        if name not in obj.__dict__:
            raise KError("Problem serializing '{}': {} is not a data member. Please follow the convention that for each of the argument listed in __init__, the layer object should have a data member with the same name.".format(obj.__class__.__name__, name))
        init_config.update({name: serialize(obj.__dict__[name])})

    return init_config


def serialize(obj):
    '''Serialize an object by memorizing its class, and its data members listed in its `__init__`.
    For complex cases (e.g., the model stucture is not an argument listed in keraflow.models.Model.__init__ ), a class should implement `pack_init_param` to perform customized behavior.
    @param obj: object.
    @return dict. The init config including keys of `class_name` and `params`.
    @see pack_init_param, keraflow.models.model.pack_init_param, keraflow.layers.base.Layer.pack_init_param
    '''

    # basic case of primitive type vairaibles
    if not hasattr(obj, '__module__'):
        return obj

    # store class name, module name for reinitialization
    class_name = obj.__class__.__name__
    module_name = obj.__module__
    init_config = {'class_name': class_name, 'module_name': module_name, 'params':{}}

    # function: store its __name__ (the class_name is always 'function', which is helpless for unserialization)
    # objcet with pack_init_param defined: store the initializing parameters returned by pack_init_param.
    # normal object: store the initializing parameters defined in __init__.

    # TODO lambda function
    if class_name=='function':
        init_config.update({'name': obj.__name__})
    elif hasattr(obj, 'pack_init_param'):
        init_config['params'].update(obj.pack_init_param())
    else:
        init_config['params'] = pack_init_param(obj)
    return init_config


def unserialize(init_config):
    '''The reverse procedure of serialize. To deal with complicated cases, a class should implement `unpack_init_param`.
    @param init_config: dict. Results generated by serialize.
    @see unserialize, keraflow.models.model.unpack_init_param, keraflow.layers.base.Layer.unpack_init_param
    '''
    # basic case of primitive type vairaibles
    if not isinstance(init_config, dict) or 'class_name' not in init_config:
        return init_config

    class_name = init_config['class_name']
    module_name = init_config['module_name']

    # function: restore from __name__
    # normal object: restore from the initializing parameters defined in init_config['params']
    # objcet with unpack_init_param defined: restore from the initializing parameters returned by unpack_init_param.
    if class_name == 'function':
        return get_from_module(module_name, init_config['name'])
    else:
        cls = get_from_module(module_name, class_name, call_constructor=False)
        params = init_config['params']
        if hasattr(cls, 'unpack_init_param'):
            params = cls.unpack_init_param(params)

        return get_from_module(module_name, class_name, **params)


def ask_overwrite(overwrite, filepath):
    if overwrite:
        return True

    if not os.path.exists(filepath):
        return False

    get_input = input
    if sys.version_info[:2] <= (2, 7):
        get_input = raw_input
    overwrite = get_input('[WARNING] {} already exists - overwrite? [y/n]'.format(filepath))

    while overwrite not in ['y', 'n']:
        overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
    if overwrite == 'n':
        return False
    print('[TIP] Next time specify overwrite=True!')
    return True

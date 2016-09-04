ALIAS = {
    'objectives':{
        'acc': 'accuracy',
        'mse': 'square_error',
        'mae': 'absolute_error',
        'mape': 'absolute_percentage_error',
        'msle': 'squared_logarithmic_error',
        'se': 'square_error',
        'ae': 'absolute_error',
        'ape': 'absolute_percentage_error',
        'sle': 'squared_logarithmic_error',
        'kld': 'kullback_leibler_divergence',
        'cosine': 'cosine_proximity',
        # Note that for the alias of `m*`, `m` stands for mean (over sample points). Keraflow objective function does not average over sample points, so the name of the function does not contain `mean`. However, to follow machine learning convention, we still provide alias starting with `m`.
    },
    'regularizers':{
        'l1':' L1',
        'l2':' L2',
        'l1l2':' L1L2'
    },
    'constraints':{
        'maxnorm':' MaxNorm',
        'nonneg':' NonNeg',
        'unitnorm':' UnitNorm'
    },
    'optimizers':{
        'sgd': 'SGD',
        'rmsprop': 'RMSprop',
        'adagrad': 'Adagrad',
        'adadelta': 'Adadelta',
        'adam': 'Adam',
        'adamax': 'Adamax'
    }
}


def get_original(module_name, alias):
    if module_name not in ALIAS:
        return alias
    elif alias in ALIAS[module_name]:
        return ALIAS[module_name][alias]
    else:
        return alias


def get_alias(module_name, original):
    return {value: key for key, value in ALIAS[module_name].items()}[original]

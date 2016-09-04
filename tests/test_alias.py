from keraflow.utils import alias


def test_get_original():
    assert alias.get_original('non_keraflow_module', 'whatever') == 'whatever'
    assert alias.get_original('objectives', 'mse') == 'square_error'
    assert alias.get_original('objectives', 'square_error') == 'square_error'


if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

from keraflow import backend as B


def test_random():
    B.seed(1234)
    print(B.eval(B.random_uniform((1,2))))

# TODO write more tests


if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

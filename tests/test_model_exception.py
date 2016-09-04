import pytest
from keraflow.layers import Input, Dense
from keraflow.models import Model
from keraflow.utils.exceptions import KeraFlowError as KError


def test_compile():
    inp1 = Input(1, batch_size=1)
    inp2 = Input(1, batch_size=2)
    d1 = Dense(1)
    d2 = Dense(1)

    def compile_model(inputs, outputs):
        model = Model(inputs, outputs)
        model.compile('sgd', 'mse')

    # input should be Input type
    with pytest.raises(KError):
        compile_model(d1, d2)

    with pytest.raises(KError):
        compile_model('whatever', d2)

    # input should be of Kensor type
    with pytest.raises(KError):
        compile_model(inp1, d2)

    # batch_size conflict
    with pytest.raises(KError):
        compile_model([inp1, inp2], d2)


def tets_fit():
    # forget to compile
    with pytest.raises(KError):
        model = Model(inp1, d1)
        model.fit([[1]], [[1]])


if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

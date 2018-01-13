import numpy as np

from keraflow.layers import Input, embeddings
from keraflow.utils.test_utils import layer_test


def test_embedding():
    vocab_size = 5
    output_dim = 3

    W = np.random.rand(vocab_size, output_dim)

    layer_test(embeddings.Embedding(vocab_size, output_dim, initial_weights=[W]),
               [[0,1,2,3,4]],
               [W])

    # test dropout, just test if it can pass...
    layer_test(embeddings.Embedding(vocab_size, output_dim, dropout=0.2),
               [[0,1,2,3,4]], test_serialization=False)

    # test Embedding's support of mask
    input1 = Input(5, dtype='int32', mask_value=0)
    emb_oup = embeddings.Embedding(vocab_size, output_dim)(input1)
    assert emb_oup.tensor._keraflow_mask is not None


if __name__ == '__main__':
    fns = globals().copy().values()
    for f in fns:
        if hasattr(f, '__name__') and f.__name__.startswith('test'):
            f()

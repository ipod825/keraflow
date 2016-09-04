from keraflow import utils

from .. import backend as B
from .base import Layer


class Embedding(Layer):
    '''Vocabulary (row) vectors looking up layer.
    - input_shape: 2D, `(nb_samples, sequence_length)`
    - output_shape: 3D, `(nb_samples, sequence_length, output_dim)`
    - parameters:
        - W: `(vocabulary_size, output_dim)`
    '''
    def __init__(self, vocabulary_size, output_dim, init='uniform', dropout=0, **kwargs):
        '''
        @param vocabulary_size: int. Size of the vocabulary.
        @param output_dim: int. Length of each vocabulary vector.
        @param init: str/function. Function to initialize trainable parameters. See @ref Initializations.md.
        @param dropout: float between 0 and 1. Fraction to drop out how many row vectors (the dropout will drop the whole row vector if it is to be dropped instead of dropping several units of the row vector.).
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).
        '''
        super(Embedding, self).__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.output_dim = output_dim
        self.init = utils.get_from_module('initializations', init)
        self.dropout = dropout

    def input_dimension(self):
        return 2

    def init_param(self, input_shape):
        W = self.init((self.vocabulary_size, self.output_dim), name='{}_W'.format(self.name))
        self.set_trainable_params('W', W)

    def output(self, x):
        if B.dtype(x) != 'int32':
            x = B.cast(x, 'int32')

        W = self.W
        if 0. < self.dropout < 1.:
            ones = B.ones((self.vocabulary_size,))
            D = B.dropout(ones, self.dropout)
            D = B.expand_dims(D)
            W *= D
        return B.gather(W, x)

    def output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def support_mask(self):
        return True

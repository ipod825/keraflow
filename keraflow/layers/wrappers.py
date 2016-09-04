import keraflow.backend as B
from keraflow.layers.base import Layer
from keraflow.layers.core import Reshape


class TimeDistributed(Layer):
    '''Wrapper for apply a layer to every temporal slice of an input.
    The input should be at least 3D. The first dimension is the batch dimension, and the second dimension is considered to be the temporal dimension.

    Example of applying a dens layer to the output of an Embedding layer:
    ~~~{.py}
    vocabulary_size = 5000
    emb_dim = 50
    model = Sequential()
    model.add(Input(None, batch_size=32))  # variable length of inputs
    model.add(Embedding(vocabulary_size, emb_dim))  # output will be of shape (batch_size, sequence_length, emb_dim)
    model.add(TimeDistributed(Dense(32)))  # output will be of shape (batch_size, sequence_length, 32)
    ~~~
    '''
    def __init__(self, layer, **kwargs):
        '''
        @param layer: The layer to be wrapped.
        @param kwargs: see [Layer.__init__](@ref keraflow.layers.base.Layer.__init__).
        '''
        super(TimeDistributed, self).__init__(**kwargs)
        self.layer = layer

    def output_shape(self, input_shape):
        layer_input_shape = (input_shape[0],) + input_shape[2:]
        layer_output_shape = self.layer.output_shape(layer_input_shape)
        return input_shape[:2] + layer_output_shape[1:]

    def output(self, x):
        input_shape = self.get_tensor_shape(x)
        timesteps = B.shape(x)[1]  # use shape tensor in case input_shape[0] is None

        x = self.embed(Reshape((-1,)+input_shape[2:], include_batch_dim=True))(x)   # (nb_sampeles*timesteps, ...)
        x = self.embed(self.layer)(x)                       # (nb_sampeles*timesteps, ...)
        output_shape = self.output_shape(input_shape)
        x = B.reshape(x, (-1, timesteps)+output_shape[2:])   # (nb_sampeles, timestep, ...)
        return x

    def input_dimension(self):
        layer_input_dimension = self.layer.input_dimension()
        if layer_input_dimension is not None:
            return layer_input_dimension+1
        return None

    def support_mask(self):
        return True

from collections import OrderedDict

import copy
from .. import backend as B
from .. import utils
from ..utils import KeraFlowError as KError
from ..utils.user_input_utils import UserInput


class Kensor(object):
    '''Wrapper for a backend tensor.
    A Kensor wraps a backend tensor and record meta data such as the tensor's shape and path from previous kensors to this kensor. A Kensor is genrated when:
    1. An Input layer is instantiated
    2. A kensor is feed to a layer (Layer.__call__).
    '''
    def __init__(self, layer, name, shape, tensor, ancestor_info={}, path=[]):
        '''
        @param layer: Layer. The layer that outputs the kensor.
        @param name: str. The name of the kensor.
        @param shape: tuple. The shape of the tensor.
        @param tensor: backend tensor. The backend tensor of the kensor.
        @param ancestor_info: dict. Info of ancestor kensors of the kensor. Used by Model to determine trainable parameters, regularizers, etc... See Layer.__call__, [Model.compile](keraflow.models.Model.compile)
        @param path: list. The path from ancestors to the kensor. Used by Model to reconstruct the tensor graph. See Layer.__call__, [Model.unpack_init_param](keraflow.models.Model.unpack_init_param)
        '''
        self.layer = layer
        self.name = name
        self.shape = shape
        self.tensor = tensor
        self.ancestor_info = ancestor_info
        self.path = path


class Layer(object):
    ''' Building block of a model.
    A Layer transform input Kensor (s) into an output Kensor. The transformation could be one-to-one or many-to-one.
    '''
    def __init__(self, name=None, trainable=True, initial_weights=None, regularizers=None, constraints=None):
        '''
        @param name: str. The name of the layer.
        @param trainable: boolean. Whether parameters of the layer will be updated during the training process.
        @param initial_weights: dict/list/single numpy array(s) or an Nd list. The initial value of the parameters of the layer. If None, a default initialization method (defined in each layer's __init__) is used.
        @param regularizers: dict/list of regularizer instance(s)/str(alias names of regurizers). The regularizers of the parameters of the layer.
        @param constraints: dict/list of constraint instance(s)/str(alias names of constraints). The constraints of the parameters of the layer.

        Trainable parameters of a Layer will be defined in a certain order (check __init__ of each layer). For assigning `initial_weights`, `regularizers`, `constraints`, the following rules apply:
        1. If a dict is passed, e.g. {'W': np.array([[1,2],[3,4]]), 'b': np.array([1,2])}, the values are assigned according to the keys (i.e. 'W', 'b'). Missing values indicates a default behavior for the corresponding parameters.
        2. If a list is passed, e.g. [np.array([[1,2],[3,4]]), np.array([1,2])]. The values are assigned to the parameters in the same order as they are defined (could be looked up in each layer's __init__). Missing values indicates a default behavior for the corresponding parameters.
        3. For `initial_weights`, passing a single numpy array (or Nd list, will be casted into numpy array) is equivalent to passing a list with a single numpy array.

        @see @ref argument_passing_summarization
        '''
        self.name = name if name else B.unique_name(self)
        self.trainable = trainable
        self.initial_weights = {} if initial_weights is None else initial_weights
        self.regularizers = {} if regularizers is None else regularizers
        self.constraints = {} if constraints is None else constraints

        self._param_initialized = False
        self.trainable_params = OrderedDict()
        self.receivers = []
        self.updates = {}
        self.embedded_layers = []
        self.embedded_id = None

    def set_trainable_params(self, *args):
        '''Register the layer's trainable parameters. Note that the order of args affect how user initialize the layer with initial_weights, regularizers, and constraints.
        @param args: arbitrary length of arguments in the pattern of ('param_name1': tensor1, 'param_name2': tensor2, ...)
        '''
        error_pattern_msg = "Trainable weights should be set in the pattern: ('name1', param1, 'name2', param2)!!"
        if len(args) % 2 == 1:
            raise KError(error_pattern_msg)  # cover

        for i in range(0,len(args),2):
            name, param = args[i], args[i+1]
            if type(name) is not str:
                raise KError(error_pattern_msg)  # cover
            if hasattr(self, name):
                raise KError("{} is a preserved variable name for Layer, pleas use another name.".format(name))
            setattr(self, name, param)
            self.trainable_params[name] = param

    def input_dimension(self):
        '''Return the expected dimension of input tensor. Recommended to implement if inheriting Layer (single input) to help keraflow detect incorrect input shape.
        @return int. Expected dimension of the input tensor.
        @see Layer.check_input_shape
        '''
        return None

    def _allow_multiple_inputs(self):
        '''Whether the layer accepts multiple input kensors. MultiInputLayer overrides this.'''
        return False

    def check_input_shape(self, input_shapes):
        '''Check if the input shape(s) are valid. Recommended to implement if inheriting MultiInputLayer (multiple inputs) to help keraflow detect incorrect input shapes.
        Default behaviors:
        1. Single input: check input dimension equal to Layer.input_dimension()
        2. Multiple inputs: check equal shape for all input kensors.
        @param input_shapes: tuple/list of tuple. The input shape(s) of input kensor(s)
        '''
        input_shapes = utils.to_list(input_shapes)
        for input_shape in input_shapes:
            assert isinstance(input_shape, tuple)

        if not self._allow_multiple_inputs():
            if len(input_shapes)>1:
                raise KError("{} does not accept multiple inputs!!".format(self.name))

            if self.input_dimension() is not None and len(input_shapes[0])!=self.input_dimension():
                raise KError("Incrroect input shape dimension for {}. Expected: {}. Given: {}".format(self.name, self.input_dimension(), len(input_shapes[0])))
        else:
            unmatch =False
            shape0 = input_shapes[0]
            for shape in input_shapes[1:]:
                if len(shape0) != len(shape):
                    unmatch =True
                for s0, s in zip(shape0, shape):
                    if s0 is not None and s0 != s:
                        unmatch = True
                        break
                if unmatch:
                    raise KError("Incrroect input shape dimension for {}. Expecting all equal shapes. Given: {}".format(self.name, input_shapes))

    def init_param(self, input_shape):
        '''Initializes trainable parameter(s)  of the layer. Inherited layer class should implement this function if the layer has trainable parameter(s). Always use Layer.set_trainable_params to register parameters.
        @param input_shape: list of tuple(s).
        '''
        pass

    def output(self, input_tensors):
        '''Calculates the output tensor of the layer given the input tensor(s). All layers must implement this function.
        @param input_tensors: backend tensor/list of backend tensors.
        @return backend tensor. The output tesor.
        '''
        input_tensors = utils.to_list(input_tensors)
        return input_tensors[0]

    def output_shape(self, input_shapes):
        '''Calculates the shape of the output tensor given the shape of the input tensor(s).
        @param input_shapes: tuple/list of tuple. The shape(s) of the input tensor(s).
        @return tuple. The shape of the output tensor.
        '''
        input_shapes = utils.to_list(input_shapes)
        return input_shapes[0]

    def support_mask(self):
        return False

    def pack_init_param(self):
        '''Check keraflow.utils.generic_utils.serialize.
        '''
        params = utils.pack_init_param(self, include_kwargs=['name', 'trainable'])

        # special care for regularizers and constraints
        name_map = {value:key for key, value in self.trainable_params.items()}
        regularizers = {}
        constraints = {}
        if self.regularizers is not None:
            regularizers = {name_map[param]: utils.serialize(regularizer) for param, regularizer in self.regularizers.items()}
        if self.constraints is not None:
            constraints = {name_map[param]: utils.serialize(constraint) for param, constraint in self.constraints.items()}

        params['regularizers'] = regularizers
        params['constraints'] = constraints

        return params

    def get_tensor_shape(self, tensor):
        '''Get the shape of a tensor. Note that the we could only get the shape if the tensor is outputted by a layer of an embedded layer.
        @param tensor: backend tensor.
        @return tuple. Shape of the tensor.
        '''
        if '_keraflow_shape' not in tensor.__dict__.keys():
            raise KError('Missing _keraflow_shape: You could only get the shape of a tensor outputted by a (embedded) keraflow layer!!')  # cover
        return tensor._keraflow_shape

    def get_tensor_mask(self, tensor):
        if '_keraflow_mask' not in tensor.__dict__.keys():
            return None
        return tensor._keraflow_mask

    def get_weights(self):
        ''' Gets trainable parameter name, value (numpy arrays) pairs. The names are decided by Layer.init_param of each layer.
        @return dict.
        '''

        names = list(self.trainable_params.keys())
        param_values = [B.eval(param) for param in self.trainable_params.values()]
        res = {name: param_value for name, param_value in zip(names, param_values)}

        embedded_layers_weights = {}
        for layer in self.embedded_layers:
            layer_weights = layer.get_weights()
            if len(layer_weights)>0:
                embedded_layers_weights[layer.embedded_id] = layer_weights

        if len(embedded_layers_weights)>0:
            res['embedded_layers_weights'] = embedded_layers_weights

        return res

    @staticmethod
    def unpack_init_param(params):
        '''Check keraflow.utils.generic_utils.unserialize.
        @param params: dict. The result from pack_init_param
        '''
        # restore the initializing parameters defined in __init__
        res = params.copy()

        # special care for regularizers and constraints
        regularizers = res.pop('regularizers')
        constraints = res.pop('constraints')
        for key, value in res.items():
            res[key] = utils.unserialize(value)

        if regularizers is not None:
            res['regularizers'] = {name: utils.unserialize(r_config) for name, r_config in regularizers.items()}
        if constraints is not None:
            res['constraints'] = {name: utils.unserialize(c_config) for name, c_config in constraints.items()}
        return res

    def embed(self, target_layer):
        '''Embeds the target layer such that the its trainable parameters (along with regularizers and constraints on the parameters) are treated as the host layer's parameters and are updated during traing process.
        Use it in a custmoized layer's output() function like:
        ~~~{.py}
        def output(self, x):
            dx = self.embed(Dense(64, regularizers=['l1']))(x)
        ~~~
        @param target_layer: Layer. The target layer.
        @return callable: take a tensor as input and return the result of output tensor transformed by the target layer.
        '''
        # The condition prevents re-embedding the same target layer when a shared layer's (self) output() get called multiple times.
        if not target_layer._param_initialized:
            target_layer.embedded_id = str(len(self.embedded_layers))
            self.embedded_layers.append(target_layer)

        def call(input_tensors):
            return target_layer._init_and_output(utils.to_list(input_tensors))

        return call

    def __call__(self, input_kensors):
        '''Feed an input kensor or multiple input kensors to the layer and outputs another Kensor.
        @param input_kensors: Kensor/list of Kensor. The input kensor(s) of the layer
        @return Kensor. An output kensor.
        '''

        input_kensors = utils.to_list(input_kensors)

        for i in range(len(input_kensors)):
            # cases like Dense(2)(Sequential([Input(32),Dense(64)]))
            if isinstance(input_kensors[i], SequentialLayer):
                input_kensors[i] = input_kensors[i].get_output_kensor()

            if not isinstance(input_kensors[i], Kensor):
                raise KError("You should only feed a layer with a Keraflow tensor. Check if you instantiate but forget to feed the layer '{}'".format(input_kensors[i].name))

            # Check graph cycle
            if input_kensors[i].layer == self:
                raise KError('Recursive feeding layer {}. Keraflow does not support recursive feeding!!'.format(self.name))

        # Establish path for serialization
        input_names = [kensor.name for kensor in input_kensors]
        out_name = '{}_{}'.format(self.name, len(self.receivers))
        input_path = utils.concat_lists([kensor.path for kensor in input_kensors])
        path = input_path + [(input_names, self.name, out_name)]

        # Sets the layer as input layers' receiver, used by model to retrieve all layers. See Model.get_layers()
        for kensor in input_kensors:
            kensor.layer.receivers.append(self)

        # Transform input tensors to an output tensor
        output_tensor = self._init_and_output([kensor.tensor for kensor in input_kensors])
        output_shape = self.get_tensor_shape(output_tensor)

        # Establish ancestor info for model to track what to update in the graph
        ancestor_info = utils.dict_unify([kensor.ancestor_info for kensor in input_kensors])
        ancestor_info[self.name] = {'trainable_params': self._get_self_and_embedded_trainable_params(),
                                    'regularizers': self._get_self_and_embedded('regularizers'),
                                    'constraints': self._get_self_and_embedded('constraints'),
                                    'updates': self._get_self_and_embedded('updates')}

        # Create and return an output kensor
        output_kensor = Kensor(self, out_name, output_shape, output_tensor, ancestor_info, path)

        return output_kensor

    def _get_self_and_embedded_trainable_params(self):
        '''Retrieve trainable parameters of the layer and all its embedded layers.
        @return list. The list of trainable params (backend tensors).
        '''
        if not self.trainable:
            return []

        res = list(self.trainable_params.values())
        for layer in self.embedded_layers:
            res += layer._get_self_and_embedded_trainable_params()
        return res

    def _get_self_and_embedded(self, name):
        '''Retrieve regularizers/constraints/updates of the layer and all its embedded layers.
        @param name: accepts 'regularizers', 'constraints', 'updates'
        @return dict. In the form of {backend tensor: regularizers/constraints/updates ops}.
        '''
        res = copy.copy(self.__dict__[name])  # copy here because we update it below.

        for layer in self.embedded_layers:
            res.update(layer._get_self_and_embedded(name))
        return res

    def _init_and_output(self, input_tensors):
        '''Initializes the layer and transformed input tensors into an output tensor.
        '''
        input_tensors = utils.to_list(input_tensors)

        input_shapes = [self.get_tensor_shape(tensor) for tensor in input_tensors]
        self.check_input_shape(input_shapes)

        # Initialize layer weights, regularizers, constraints
        if not self._param_initialized:
            self.init_param(utils.unlist_if_one(input_shapes))
            # Function to get weights/regularizers/constraints
            self.get_wrc = UserInput(list(self.trainable_params.keys()))
            self._set_weights(self.initial_weights)
            self._set_regularizers(self.regularizers)
            self._set_constraints(self.constraints)

            del self.initial_weights

            self._param_initialized = True

        output_shape = utils.to_tuple(self.output_shape(utils.unlist_if_one(input_shapes)))
        output_tensor = self.output(utils.unlist_if_one(input_tensors))

        # Embed shape into tensor for easy access by Layer.get_tensor_shape()
        self._set_tensor_shape(output_tensor, output_shape)

        # Embed the input mask into the output tensor (if self supports masking)
        mask = None
        if self.support_mask():
            for tensor in input_tensors:
                if self.get_tensor_mask(tensor) is not None:
                    if mask is not None:
                        raise KError("Supports only single mask from inputs!!")
                    else:
                        mask = self.get_tensor_mask(tensor)
        self._set_tensor_mask(output_tensor, mask)

        return output_tensor

    def _set_tensor_shape(self, tensor, shape):
        tensor._keraflow_shape = shape

    def _set_tensor_mask(self, tensor, mask):
        tensor._keraflow_mask = mask

    def _set_weights(self, weights):
        '''Sets the weights of the layer.
        @param weights: a dict/list of Numpy arrays.
        @see Layer.__init__
        '''
        if isinstance(weights, dict) and 'embedded_layers_weights' in weights:
            embedded_layers_weights = weights.pop('embedded_layers_weights')
            # Embedded layers are not saved when serializing models.
            # Hence the name of reconstructed embedded layers are not guaranteed to be the same (if auto generated).
            # Thererfore, we use the order of the embedded_layers as their unique id to memorize their weights.
            for layer in self.embedded_layers:
                if layer.embedded_id in embedded_layers_weights:
                    layer._set_weights(embedded_layers_weights[layer.embedded_id])

        weights_list = self.get_wrc(weights, array=True, allow_missing='skip', error_context='initial weights of '+self.name)
        params = [self.trainable_params[w[0]] for w in weights_list]
        names = [w[0] for w in weights_list]
        weights = [w[1] for w in weights_list]

        for p, w, name in zip(params, weights, names):
            param_shape = B.eval(p).shape
            if param_shape != w.shape:
                raise KError("Layer weight shape mismatch for '{}' of '{}'!! Expected: {}. Given: {}.".format(name, self.name, param_shape, w.shape))
            B.set_value(p, w)

    def _set_regularizers(self, regularizers):
        '''Sets the regularizers of the layer.
        @param regularizers: dict/list of regulizer object(s)/str (name of regularizer(s)) objects.
        @see Layer.__init__
        @see keraflow.regularizers
        '''
        regularizers_list = self.get_wrc(regularizers, allow_missing='fill_none', error_context='regularizers of '+self.name)
        self.regularizers = {}
        for name, regularizer in regularizers_list:
            if regularizer is not None:
                self.regularizers[self.trainable_params[name]] = regularizer

    def _set_constraints(self, constraints):
        '''Sets the constraints of the layer.
        @param constraints: dict/list of constraint objects(s)/str (name of constraint(s)).
        @see Layer.__init__
        @see keraflow.constraints
        '''
        constraints_list = self.get_wrc(constraints, allow_missing='fill_none', error_context='constraints of '+self.name)
        self.constraints = {}
        for name, constraint in constraints_list:
            if constraint is not None:
                self.constraints[self.trainable_params[name]] = constraint


class MultiInputLayer(Layer):
    '''Base class for layer accepting multiple inputs.
    '''
    def _allow_multiple_inputs(self):
        return True


class Input(Layer, Kensor):
    '''The entering point of each model.
    '''
    def __init__(self, shape, dtype=B.floatx(), batch_size=None, mask_value=None, **kwargs):
        '''
        @param shape: tuple. The expected shape of the input data.
        @param dtype: str/type. the expected data type of the input data.
        @param batch_size: int. The expected batch size. If None, accepts any number of batch. Should be same for all inputs of a model.
        @param mask_value: int. The input value to be masked. If not None, a special mask tensor indicating which indices in the input should be skipped will be passed to the following layers (if the layers support masking). This is majorly used for recurrent layers, which may take variable length input.
        @param kwargs: see Layer.__init__
        '''
        if isinstance(dtype, int):
            raise KError("Input.__init__:You forget to round the shape argument with parathness, e.g. Input(shape=(1,2))")
        shape = (batch_size,)+utils.to_tuple(shape)
        Layer.__init__(self, **kwargs)
        Kensor.__init__(self,
                        layer=self,
                        name=self.name,
                        shape=shape,
                        tensor=B.placeholder(shape, dtype=dtype, name=self.name))
        self._set_tensor_shape(self.tensor, shape)
        if mask_value is not None:
            assert len(shape)==2
            self._set_tensor_mask(self.tensor, B.not_equal(self.tensor, mask_value))
        else:
            self._set_tensor_mask(self.tensor, None)

        self.dtype = dtype
        self.batch_size = batch_size
        self.mask_value = mask_value

    def pack_init_param(self):
        params = utils.pack_init_param(self, include_kwargs=['name'])
        params['shape'] = params['shape'][1:]
        return params

    @staticmethod
    def unpack_init_param(params):
        return params

    def __call__(self, input_tensor):
        raise KError("You should not feed anything to an Input Layer!!")


class SequentialLayer(Layer):
    '''Class for making Sequential a Layer. This class is inherited by Sequential. Use Sequential instead of this class.'''
    def __init__(self, layers=[], **kwargs):
        '''
        @param layers: list of Layer.
        @param kwargs: see Layer.__init__
        '''
        super(SequentialLayer, self).__init__(**kwargs)
        if not isinstance(layers, list):
            raise KError("A Sequential should be initialized with a list of layers")

        self.layers = list(layers)
        self._cached_output_kensor = None

    def add(self, layer):
        '''Append a layer to the layers stack.'''
        self.layers.append(layer)

    def get_output_kensor(self):
        '''Get the output_kensor. This requires the first layer be an Input.'''
        if self._cached_output_kensor is not None:
            return self._cached_output_kensor

        if not isinstance(self.layers[0], Input):
            raise KError("To access the output kensor of a SequentialLayer without feeding it. The first inner layer should be an Input layer!!")

        output_kensor = self.layers[0]
        for layer in self.layers[1:]:
            output_kensor = layer(output_kensor)

        return output_kensor

    def output(self, input_tensors):
        output_tensor = input_tensors
        for layer in self.layers:
            output_tensor = self.embed(layer)(output_tensor)
        return output_tensor

    def output_shape(self, input_shapes):
        res = input_shapes
        for layer in self.layers:
            res = layer.output_shape(res)
        return res

    def check_input_shape(self, input_shapes):
        self.layers[0].check_input_shape(input_shapes)

    def pack_init_param(self):
        '''Check keraflow.utils.generic_utils.serialize.
        '''
        return {'name': self.name,
                'layers': [utils.serialize(layer) for layer in self.layers]}

    @staticmethod
    def unpack_init_param(init_config):
        '''Check keraflow.utils.generic_utils.unserialize.
        @param init_config: dict. The result from pack_init_param
        '''
        return {'name': init_config['name'],
                'layers': [utils.unserialize(layer_init_config) for layer_init_config in init_config['layers']]}

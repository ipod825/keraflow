
import sys
from collections import OrderedDict

import numpy as np

from tqdm import tqdm

from . import backend as B
from . import utils
from .layers.base import Input, Kensor, Layer, SequentialLayer
from .utils import KeraFlowError as KError
from .utils.alias import get_alias
from .utils.user_input_utils import UserInput


class Model(object):
    '''A model is a directed [Kensor](keraflow.layers.base.Kensor) graph. [Kensor](keraflow.layers.base.Kensor) connects with one another by [Layer.__call__](keraflow.layers.base.Layer.__call__) and users specify the input kensors and out kensors to build the graph.
    @see @ref an_over_view_of_model
    '''
    def __init__(self, inputs, outputs, name=None):
        '''
        @param inputs: single/list of [Input](@ref keraflow.layers.base.Input).
        @param outputs: single/list of [Kensor](@ref keraflow.layers.base.Kensor).
        @param name: str. Name of the model for easy debugging.
        '''
        input_kensors = utils.to_list(inputs)
        output_kensors = utils.to_list(outputs)

        for i, kensor in enumerate(input_kensors):
            if issubclass(kensor.__class__, SequentialLayer):
                input_kensors[i] = kensor.layers[0]

        self.input_kensors = input_kensors
        self.output_kensors = output_kensors
        self.name = name if name else B.unique_name(self)

    def compile(self, optimizer, loss, metrics=[]):
        '''Configure the model and prepare inner utilized tensors.
        @param optimizer: str(name of optimizer class)/optimizer object. See keraflow.optimizers
        @param loss: dict/list/single objective function(s)/str(name of objective function(s)). Objective(s) for output(s). See @ref Objectives.md for a list of predefined objective functions.
        @param metrics: dict/list/single objective function(s)/str(name of objective function(s)). Objective(s) for outputs(s). See @ref Objectives.md for a list of predefined objective functions.
        '''
        if not hasattr(self, 'compiled'):
            self.compiled = True
        else:
            return

        # Check input/output correctness
        for kensor in self.input_kensors:
            if not isinstance(kensor, Input):
                if isinstance(kensor, Kensor) or isinstance(kensor, Layer):
                    raise KError("Input kensor '{}' of '{}' should be of Input type.".format(kensor.name, self.name))
                else:
                    raise KError("All input ports of '{}' should be of Input type.".format(self.name))
        for kensor in self.output_kensors:
            if not isinstance(kensor, Kensor):
                raise KError("Output kensor '{}' should be of Kensor type. You might forget to feed '{}'".format(kensor.name, kensor.name))

        batch_size = self.input_kensors[0].batch_size
        for kensor in self.input_kensors:
            if batch_size != kensor.batch_size:
                raise KError("Mismatch batch size: {}, {}", batch_size, kensor.batch_size)
        self.batch_size = batch_size

        # Setup input/output tensors
        self.input_tensors = [kensor.tensor for kensor in self.input_kensors]
        self.output_tensors = [kensor.tensor for kensor in self.output_kensors]
        self.input_names = [kensor.layer.name for kensor in self.input_kensors]
        self.output_names = [kensor.layer.name for kensor in self.output_kensors]
        self.input_shapes = [utils.to_tuple(kensor.shape) for kensor in self.input_kensors]
        self.output_shapes = [utils.to_tuple(kensor.shape) for kensor in self.output_kensors]
        self.get_inp = UserInput(self.input_names)
        self.get_oup = UserInput(self.output_names)
        self.optimizer = utils.get_from_module('optimizers', optimizer)
        self.target_tensors = [B.placeholder(shape, name=name+ '_target')
                               for shape, name in zip(self.output_shapes, self.output_names)]

        # Prepare trainable_params, regularizers, constraints
        layers_info = utils.dict_unify([kensor.ancestor_info for kensor in self.output_kensors])
        self.trainable_params = []
        self.regularizers = {}
        self.constraints = {}
        self.layer_updates = []
        for info in layers_info.values():
            self.trainable_params += info['trainable_params']
            self.regularizers.update(info['regularizers'])
            self.constraints.update(info['constraints'])
            if len(info['updates'])>0:
                self.layer_updates+=info['updates']

        # Prepare loss/metric tensors. Calculating total loss requires sample_weights and is thus postponed to _compile_loss.
        loss_list = self.get_oup(loss, allow_missing=False, no_single_value=False, error_context='objectives of '+self.name)
        self.loss_fns = [utils.get_from_module('objectives', l) for l in loss_list]

        metric_list = self.get_oup(metrics, allow_missing='fill_none', error_context='metrics of '+self.name)
        self.metric_tensors = []
        self.metric_labels = []
        for i, (output_name, metric_fn) in enumerate(metric_list):
            if metric_fn is not None:
                fn = utils.get_from_module('objectives', metric_fn)
                self.metric_tensors.append(B.mean(fn(self.output_tensors[i], self.target_tensors[i])))
                self.metric_labels.append('{}'.format(get_alias('objectives', fn.__name__)))
                if len(self.output_kensors)!=1:
                    self.metric_labels[-1] += '({})'.format(output_name)

        # The following functions are compiled on demand.
        self._train_fn = None
        self._evaluate_fn= None
        self._predict_fn = None

    def _compile_loss(self, sample_weights):
        '''Final step for preparing loss, metrics tensors and training updates for further compilation.
        @param sample_weights: list/numpy array. Weights of each sample.
        '''
        if not hasattr(self, 'loss_compiled'):
            self.loss_compiled = True
        else:
            return

        if sample_weights is not None:
            weight_tensor = B.placeholder(ndim=1, name=self.name + '_sample_weights')
            self.sample_weight_tensors = [weight_tensor]
        else:
            weight_tensor = None
            self.sample_weight_tensors = []

        output_losses = []
        for loss_fn, y_pred, y_true, name in zip(self.loss_fns, self.output_tensors, self.target_tensors, self.output_names):
            # calculate per sampe loss (average over sample dimensions)
            loss_array = loss_fn(y_pred, y_true)
            score_array = B.mean(loss_array, axis=list(range(1, B.ndim(loss_array))))

            # weighting per sample loss
            if weight_tensor is not None:
                score_array *= weight_tensor
                # average only on non zero weight
                score_array /= B.mean(B.cast(B.not_equal(weight_tensor, 0), B.floatx()))

            # average over smples
            output_losses.append(B.mean(score_array))

        output_loss = sum(output_losses)
        regularizer_losses = [regularizer(param)
                              for param, regularizer in self.regularizers.items()]
        regularizer_loss = sum(regularizer_losses)
        train_loss = output_loss + regularizer_loss
        test_loss = output_loss
        loss_tensor = B.in_train_phase(train_loss, test_loss)

        # Treat loss as one of the metrics
        self.metric_tensors.insert(0, loss_tensor)
        self.metric_labels.insert(0, 'loss')

        # Prepare updates
        raw_updates = self.optimizer.get_updates(loss_tensor, self.trainable_params)
        self.training_updates = []
        for p, new_p in raw_updates:
            if p in self.constraints:
                new_p = self.constraints[p](new_p)
            self.training_updates.append((p, new_p))

    def fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], sample_weights=None, validation_split=0.0, validation_data=None, shuffle=True):
        '''Adjust model parameters to fit input data and true answers.
        @param x: dict/list/single numpy array(s) or an Nd list. Input data.
        @param y: dict/list/single numpy array(s) or an Nd list. True answer.
        @param batch_size: int. Number of samples per gradient update. Report error if unequal to (not `None`) input kensors' `batch_size`.
        @param nb_epoch: int. The number of epochs to train the model.
        @param verbose: 0 for no logging, 1 for logging to stdout.
        @param callbacks: list of keraflow.callbacks.Callback instances.
        @param sample_weights: list/numpy array. Weights of each sample.
        @param validation_split: float between 0 and 1.  Fraction of the training data to be used as validation data.
        @param validation_data: tuple (`val_x`, `val_y`), or (`val_x`, `val_y`, `val_sample_weight`) to be used as held-out validation data. `val_x`, `val_y`, and `val_sample_weights` accept the same type of data as `x`, `y`, and `sample_weights` respectively. This argument override `validation_split` argument.
        @param shuffle: boolean. Whether to shuffle the samples at each epoch.
        @return list of dict. Each dict in the list describes the loss (and validation loss if applicable) and metrics for one epoch.

        @note
        Rules for assigning x, y:
        1. If a dict is passed (and the corresponding input kensors and output layers are named), the values are assigned according to the keys.
        2. If a list of numpy array is passed, the arrays are assigned to the kensor in the order you construct the model.
        3. If there's only a single input/output kensor, a single numpy array (or Nd list, will be casted into numpy array) is acceptable.
        '''
        if not hasattr(self, '_train_fn'):
            raise KError("Please call compile before calling fit/predict/eval functions")

        if self.batch_size is not None and batch_size!=self.batch_size:
            raise KError("Batch size of {} is fixed to {}. Passed mismatched value: {}".format(self.name, self.batch_size, batch_size))

        # prepare training data
        numpy_inputs = self._prepare_io_numpy_arrays(x, y, sample_weights) + [B.train_phase()]

        if self._train_fn is None:
            self._compile_loss(sample_weights)
            tensor_inputs = self.input_tensors + self.target_tensors + self.sample_weight_tensors + [B.learning_phase()]
            self._train_fn = B.function(tensor_inputs,
                                        self.metric_tensors,
                                        updates=self.training_updates+self.layer_updates)

        # prepare validation data
        val_x = val_y = val_sample_weights = None
        val_numpy_inputs = None
        if validation_data is not None:
            if not isinstance(validation_data, tuple):
                raise KError("validation_data accepts only tuple!!")
            if len(validation_data)<2:
                raise KError("validation_data format (val_x, val_y, [val_sample_weights])")
            val_x, val_y = validation_data[:2]
            if len(validation_data) == 3:
                val_sample_weights = validation_data[2]

            val_x = self.get_inp(val_x, array=True, allow_missing=False, error_context='input validation arrays of '+ self.name)
            val_y = self.get_oup(val_y, array=True, allow_missing=False, error_context='output validation arrays of' + self.name)

            val_numpy_inputs = self._prepare_io_numpy_arrays(val_x, val_y, val_sample_weights) + [B.test_phase()]
        elif 0. < validation_split < 1.:
            nb_sample = len(numpy_inputs[0])
            ori_numpy_inputs = numpy_inputs[:-1]  # last one is training phase
            numpy_inputs = []
            val_numpy_inputs = []
            s = int(nb_sample * (1. - validation_split))
            for array in ori_numpy_inputs:
                numpy_inputs.append(self._slice_arrays(array, 0, s))
                val_numpy_inputs.append(self._slice_arrays(array, s))

            numpy_inputs.append(B.train_phase())
            val_numpy_inputs.append(B.test_phase())
            val_x = val_numpy_inputs[:len(self.input_tensors)]
            val_y = val_numpy_inputs[len(self.input_tensors):len(self.input_tensors)+len(self.output_tensors)]
            if sample_weights is not None:
                val_sample_weights = val_numpy_inputs[-2]

        # compile the _evaluate_fn (for validating val data later) by calling evaluate.
        # Use only one sample to save computation.
        if val_numpy_inputs is not None:
            if val_sample_weights is None:
                self.evaluate([a[:1] for a in val_x], [a[:1] for a in val_y], val_sample_weights)
            else:
                self.evaluate([a[:1] for a in val_x], [a[:1] for a in val_y], val_sample_weights[:1])

        # Run training loop
        nb_sample = len(numpy_inputs[0])
        batches = self._make_batchs(nb_sample, batch_size)
        history = []
        index_array = np.arange(nb_sample)
        for cbk in callbacks:
            cbk._set_model(self)

        for epoch in range(nb_epoch):
            if shuffle:
                np.random.shuffle(index_array)

            batch_logs = []
            epoch_msg = "Epoch {:>3} ".format(epoch)
            for batch_ind, (start, stop) in enumerate(tqdm(batches, desc=epoch_msg, unit='batch', leave=False)):
                batch_ids = index_array[start:stop]
                batch = self._slice_arrays(numpy_inputs[:-1], batch_ids) + [numpy_inputs[-1]]
                batch_out = self._train_fn(batch)
                batch_log = OrderedDict({l: float(m) for l, m in zip(self.metric_labels, batch_out)})
                for cbk in callbacks:
                    cbk.on_batch_end(batch_ind, batch_log)
                batch_logs.append(batch_log)

            epoch_log = utils.average_dicts(batch_logs)

            if val_numpy_inputs is not None:
                val_out = self._evaluate_loop(val_numpy_inputs, batch_size=batch_size)
                for l, m in zip(self.metric_labels, val_out):
                    epoch_log['val_'+ l] = float(m)

            if verbose>0:
                msg = epoch_msg+':'
                for label, res in epoch_log.items():
                    msg += ' {}: {:.4f}'.format(label, res)
                print(msg)
            for cbk in callbacks:
                cbk.on_epoch_end(epoch, epoch_log)
            history.append(epoch_log)

        for cbk in callbacks:
            cbk.on_train_end(epoch, epoch_log)

        return history

    def _make_batchs(self, nb_sample, batch_size):
        nb_batch = int(np.ceil(nb_sample/float(batch_size)))
        batches = [(i * batch_size, min(nb_sample, (i + 1) * batch_size))
                   for i in range(0, nb_batch)]
        return batches

    def _slice_arrays(self, arrays, start=None, stop=None):
        '''Slice and return the input array(s)
        @param arrays: single/list of numpy arrays or an Nd list.
        @param start: int. Starting point of the slice.
        @param stop: int. Stopping point of the slice.
        @return single/list of numpy arrays.
        '''
        if isinstance(arrays, list):
            if isinstance(start, int):
                return [x[start:stop] for x in arrays]
            else:
                return [x[start] for x in arrays]
        else:
            if isinstance(start, int):
                return arrays[start:stop]
            else:
                return arrays[start]

    def predict(self, x, batch_size=32, train_mode=False):
        '''Returns the output values given the input data.
        @param x: dict/list/single numpy array(s) or an Nd list. Input data. Rules for assigning `x` is the same as in [fit](@ref Model.fit).
        @param batch_size: int. The batch size to predict the testing data. Might cause memorage error if set too large.
        @param train_mode: boolean. For debugging usage. Do not use this flag in your code.
        @return list of numpy arrays. The length of the list is equal to the number of output channels.
        '''
        if not hasattr(self, '_predict_fn'):
            raise KError("Please call compile before calling fit/predict/eval functions")

        np_learning_phase = B.train_phase() if train_mode else B.test_phase()
        numpy_inputs = self._prepare_io_numpy_arrays(x)+[np_learning_phase]

        if self._predict_fn is None:
            tensor_inputs = self.input_tensors + [B.learning_phase()]
            # Note that learning phase is in the output list. This prevents theano warning that learning phase is not used when no layer use learning phase.
            self._predict_fn = B.function(tensor_inputs,
                                          self.output_tensors+[B.cast(B.learning_phase(), B.floatx())],
                                          updates=self.layer_updates)
        # return self._predict_fn(numpy_inputs)[:-1]
        batches = self._make_batchs(len(numpy_inputs[0]), batch_size)
        outputs = [list() for _ in range(len(self.output_kensors))]
        for start, stop in batches:
            batch = self._slice_arrays(numpy_inputs[:-1], start, stop) + [numpy_inputs[-1]]
            batch_outputs = self._predict_fn(batch)[:-1]
            for i in range(len(outputs)):
                outputs[i].append(batch_outputs[i])

        for i in range(len(outputs)):
            outputs[i] = np.concatenate(outputs[i], axis=0)

        return np.asarray(outputs)

    def evaluate(self, x, y, sample_weights=None, batch_size=32, train_mode=False):
        '''Returns the loss value and metrics values given the input data and true answer.
        @param x: dict/list/single numpy array(s) or an Nd list. Input data. Rules for assigning `x` is the same as in [fit](@ref Model.fit).
        @param y: dict/list/single numpy array(s) or an Nd list. True answer. Rules for assigning `y` is the same as in [fit](@ref Model.fit).
        @param sample_weights: list/numpy array. Weights of each sample.
        @param batch_size: int. The batch size to predict the testing data. Might cause memorage error if set too large.
        @param train_mode: boolean. For debugging usage. Do not use this flag in your code.
        '''
        if not hasattr(self, '_evaluate_fn'):
            raise KError("Please call compile before calling fit/predict/eval functions")

        np_learning_phase = B.train_phase() if train_mode else B.test_phase()
        numpy_inputs = self._prepare_io_numpy_arrays(x, y, sample_weights) + [np_learning_phase]

        if self._evaluate_fn is None:
            self._compile_loss(sample_weights)
            tensor_inputs = self.input_tensors + self.target_tensors + self.sample_weight_tensors + [B.learning_phase()]
            self._evaluate_fn = B.function(tensor_inputs,
                                           self.metric_tensors,
                                           updates=self.layer_updates)

        return self._evaluate_loop(numpy_inputs, batch_size)

    def _evaluate_loop(self, numpy_inputs, batch_size=32):
        nb_sample = len(numpy_inputs[0])
        batches = self._make_batchs(nb_sample, batch_size)
        outputs = [0.]*len(self.metric_tensors)
        for start, stop in batches:
            batch = self._slice_arrays(numpy_inputs[:-1], start, stop) + [numpy_inputs[-1]]
            batch_outputs = self._evaluate_fn(batch)
            for i in range(len(outputs)):
                outputs[i] += batch_outputs[i]*(stop-start)

        for i in range(len(outputs)):
            outputs[i] /= nb_sample

        return np.asarray(outputs)

    def _prepare_io_numpy_arrays(self, x=None, y=None, sample_weights=None):
        res = []
        if x is not None:
            x = self.get_inp(x, array=True, allow_missing=False, error_context='input arrays of '+ self.name)
            self._validate_io_arrays_shapes(self.input_names, x, self.input_shapes)
            res += x
        if y is not None:
            y = self.get_oup(y, array=True, allow_missing=False, error_context='output arrays of' + self.name)
            self._validate_io_arrays_shapes(self.output_names, y, self.output_shapes)
            res += y

        if sample_weights is not None:
            res.append(np.asarray(sample_weights))
        return res

    def _validate_io_arrays_shapes(self, names, arrays, shapes):
        # make sure the arrays are 2D
        for i in range(len(arrays)):
            if len(arrays[i].shape)==1:
                arrays[i] = np.expand_dims(arrays[i], 1)

        for array, name, shape in zip(arrays, names, shapes):
            if len(array.shape) != len(shape):
                raise KError('Input dimension mismatch for {}. Expected: {} (batch dimension included). Given: {}'.format(name, len(shape), len(array.shape)))
            for a, p in zip(array.shape, shape):
                if p is not None and p != a:
                    raise KError('Input shape mismatch for {}. Expected: {}. Given: {}'.format(name, shape, array.shape))

    def save_to_file(self, arch_fname, weight_fname=None, overwrite=False, **kwargs):
        '''Save the model to disk for later use. Combining operations done by save_arch and save_weights.
        @param arch_fname: str. The file name for storing the model architecture. Should ends with either 'json' or 'yaml' for different storing types.
        @param weight_fname: str. The file name for storing weights of the model. If None, the weights are not saved.
        @param overwrite: Whether to overwrite existing files.
        @param kwargs: keyword arguments. The keyword arguments for `json.dump` or `yaml.dump`.
        '''
        self.save_arch(arch_fname, overwrite=overwrite, **kwargs)

        if weight_fname is not None:
            self.save_weights(weight_fname, overwrite=overwrite)

    def save_arch(self, arch_fname, overwrite=False, **kwargs):
        '''
        @param arch_fname: str. The file name for storing the model architecture. Should ends with either 'json' or 'yaml' for different storing types.
        @param kwargs: keyword arguments. The keyword arguments for `json.dump` or `yaml.dump`.
        @param overwrite: Whether to overwrite existing files.
        '''
        import importlib
        if arch_fname[-5:]=='.json':
            dumper = importlib.import_module('json')
        elif arch_fname[-5:]=='.yaml':
            dumper = importlib.import_module('yaml')
        else:
            raise KError('Please specify filename extension (json/yaml)')

        if not utils.ask_overwrite(overwrite, arch_fname):
            return

        init_config = utils.serialize(self)
        with open(arch_fname, 'w') as f:
            dumper.dump(init_config, f, **kwargs)

    def save_weights(self, weight_fname, overwrite=False):
        '''
        @param weight_fname: str. The file name for storing weights of the model. If None, the weights are not saved.
        @param overwrite: Whether to overwrite existing files.
        '''
        if not utils.ask_overwrite(overwrite, weight_fname):
            return

        import warnings
        res = {}
        layers = self._get_layers()
        for name, layer in layers.items():
            weights = layer.get_weights()
            if len(weights)>0:
                res[name] = weights

        if len(res)==0:
            warnings.warn("There is no parameters in {}. Skip saving!!".format(self.name))
        else:
            if sys.version_info.major==2:
                import hickle
                hickle.dump(res, weight_fname, mode='w', compression='gzip')
            else:
                import io
                import pickle
                with io.open(weight_fname, 'wb') as f:
                    pickle.dump(res, f)

    @staticmethod
    def load_from_file(arch_fname, weight_fname=None):
        '''Load model from disk.
        @param arch_fname: str. The file name storing the model architecture.
        @param weight_fname: str. The file name storing weights of the model. If None, the weights are not loaded.
        @return Model. The Model instance reconstructed from the file. Note that you should compile it before using it.
        '''
        import importlib
        if arch_fname[-5:]=='.json':
            loader = importlib.import_module('json')
        elif arch_fname[-5:]=='.yaml':
            loader = importlib.import_module('yaml')
        else:
            raise KError('arch_fname should ends with "json" or "yaml")')

        with open(arch_fname, 'r') as f:
            init_config = loader.load(f)
            model = utils.unserialize(init_config)
            if isinstance(model, Sequential):
                model.input_kensors = [model.layers[0]]
                model.output_kensors = [model.get_output_kensor()]

        if weight_fname is not None:
            try:
                import hickle
                weights = hickle.load(weight_fname)
            except:
                import io
                import pickle
                with io.open(weight_fname, 'rb') as f:
                    weights = pickle.load(f)

            layers = model._get_layers()
            for name, layer_weights in weights.items():
                layers[name]._set_weights(layer_weights)

        return model

    def _get_layers(self):
        from collections import deque
        layers = OrderedDict()
        layer_queue = deque(self.input_kensors)
        while len(layer_queue)>0:
            layer = layer_queue.popleft()
            layer_queue.extend(layer.receivers)
            if layer.name not in layers:
                layers[layer.name] = layer
        return layers

    def pack_init_param(self):
        layers = self._get_layers()
        return {'name':self.name,
                'layers':[utils.serialize(layer) for layer in layers.values()],
                'path': utils.concat_lists([kensor.path for kensor in self.output_kensors]),
                'output_names': [kensor.name for kensor in self.output_kensors]}

    @staticmethod
    def unpack_init_param(init_config):
        input_kensors = []
        layers = {}
        tensors = {}
        # reinitialize all layers
        for layer_init_config in init_config['layers']:
            layer = utils.unserialize(layer_init_config)
            layers[layer.name] = layer
            if isinstance(layer, Input):
                input_kensors.append(layer)
                tensors[layer.name] = layer

        # reconnected all layers
        for iname, lname, oname in init_config['path']:
            output_port = layers[lname]([tensors[name] for name in iname])
            tensors[oname] = output_port

        output_kensors = [tensors[name] for name in init_config['output_names']]

        return {'name': init_config['name'],
                'inputs': input_kensors,
                'outputs': output_kensors}


class Sequential(Model, SequentialLayer):
    '''Model with single input and single output. Also severs a Layer (inheriting [SequentialLayer](@ref keraflow.layers.base.SequentialLayer)) for squeezing a sequence of layer into a single one.
    '''
    def __init__(self, layers=[], **kwargs):
        '''
        @param layers: a list of layer instance.
        @param kwargs: see Layer.__init__. This is the common args for Layer. For Sequential, you could set `name` and `trainable`.
        '''
        SequentialLayer.__init__(self, layers, **kwargs)
        self._as_model = False

    def compile(self, optimizer, loss, metrics=[]):
        '''Configure the model and prepare inner utilized tensors.
        @param optimizer: str(name of optimizer class)/optimizer object. See keraflow.optimizers
        @param loss: objective function/str(name of objective function). Objective for the output. See @ref Objectives.md for a list of predefined objective functions.
        @param metrics: objective function/str(name of objective function). Extra objective for the output. See @ref Objectives.md for a list of predefined objective functions.
        '''
        self.input_kensors = [self.layers[0]]
        self.output_kensors = [self.get_output_kensor()]
        Model.compile(self, optimizer, loss, metrics)
        self._as_model = True

    def predict(self, x, batch_size=32, train_mode=False):
        '''Returns the output values given the input data.
        @param x: numpy array/list. Input data.
        @param batch_size: int. The batch size to predict the testing data. Might cause memorage error if set too large.
        @param train_mode: boolean. For debugging usage. Do not use this flag in your code.
        @return numpy array. Output value.
        '''
        return super(Sequential, self).predict(x, batch_size=batch_size, train_mode=train_mode)[0]

    def pack_init_param(self):
        if self._as_model:
            return Model.pack_init_param(self)
        else:
            return SequentialLayer.pack_init_param(self)

    @staticmethod
    def unpack_init_param(init_config):
        return {'name': init_config['name'],
                'layers': [utils.unserialize(layer_init_config) for layer_init_config in init_config['layers']]}

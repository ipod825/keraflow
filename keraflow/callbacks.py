import warnings

import numpy as np

from . import backend as B


class Callback(object):
    '''Abstract base class used to build new callbacks.

    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.

    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:

        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    '''
    def _set_params(self, params):
        self.params = params

    def _set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass


class ModelCheckpoint(Callback):
    '''Save the model after every epoch.
    '''
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_on_improvement=False, mode='auto'):
        '''
        @param filepath: str. Path to save the model weights.
        @param monitor: str. Quantity to monitor.
        @param verbose: verbosity mode, 0 or 1.
        @param save_on_improvement: boolean. If true, only saves the model when monitored quantity improves.
        @param mode: one of {auto, min, max}. Used only if `save_on_improvement` is true. Decides if the monitored quantity improves. If set to `max`, increase of the quantity indicates improvement, and vice versa. If set to 'auto', behaves like 'max' if `monitor` contains substring 'acc'. Otherwise, behaves like 'min'.
        '''

        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_on_improvement = save_on_improvement

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_on_improvement:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with {} available, skipping.'.format(self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch {}: {} improved from {.5f} to {.5f}, saving model to {}'.format(epoch, self.monitor, self.best, current, filepath))
                    self.best = current
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('Epoch {}: {} did not improve'.format(epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch {}: saving model to {}'.format(epoch, filepath))
            self.model.save_weights(filepath, overwrite=True)


class EarlyStopping(Callback):
    '''Stop training when a monitored quantity has stopped improving.
    '''
    def __init__(self, monitor='val_loss', patience=0, verbose=0, mode='auto'):
        '''
        @param monitor: str. Quantity to monitor.
        @param patience: number of epochs with no improvement after which training will be stopped.
        @param verbose: verbosity mode, 0 or 1.
        @param mode: one of {auto, min, max}. Decides if the monitored quantity improves. If set to `max`, increase of the quantity indicates improvement, and vice versa. If set to 'auto', behaves like 'max' if `monitor` contains substring 'acc'. Otherwise, behaves like 'min'.
        '''
        super(Callback, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires {} available!'.format(self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('Epoch {}: early stopping'.format(epoch))
                self.model.stop_training = True
            self.wait += 1


class LearningRateScheduler(Callback):
    '''Learning rate scheduler.
    '''
    def __init__(self, schedule):
        '''
        @param schedule: callable. A function that takes an epoch index as input (integer, indexed from 1) and returns a new learning rate.
        '''
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_end(self, epoch, logs={}):
        assert hasattr(self.model.optimizer, 'lr'), 'Optimizer must have a "lr" attribute.'
        lr = self.schedule(epoch)
        assert type(lr) == float, 'The output of the "schedule" function should be float.'
        B.set_value(self.model.optimizer.lr, lr)

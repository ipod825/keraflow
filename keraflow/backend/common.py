import numpy as np

_FLOATX = 'float32'
_EPSILON = 10e-8


class Backend(object):
    def __init__(self, name=None, epsilon=10e-8, floatx=_FLOATX, **kwargs):
        global _FLOATX, _EPSILON

        _EPSILON = epsilon
        _FLOATX = floatx
        self._name = name
        self._test_phase = 0
        self._train_phase = 1
        self._learning_phase = self.placeholder(shape=None, dtype='uint8', name='backend_learning_phase_input')
        self.uid_prefixes = {}
        self._seed = None

    def name(self):
        return self._name

    def epsilon(self):
        return _EPSILON

    def floatx(self):
        return _FLOATX

    def test_phase(self):
        return self._test_phase

    def train_phase(self):
        return self._train_phase

    def learning_phase(self):
        return self._learning_phase

    def seed(self, s):
        self._seed = s
        np.random.seed(self._seed)
        self.reset_random_state()

    def cast_to_floatx(self, x):
        '''Cast a Numpy array to floatx.
        '''
        return np.asarray(x, dtype=_FLOATX)

    def unique_name(self, x):
        class_name = x.__class__.__name__.lower()
        if class_name not in self.uid_prefixes:
            self.uid_prefixes[class_name] = 1
        else:
            self.uid_prefixes[class_name] += 1
        return class_name+str(self.uid_prefixes[class_name])

    def in_train_phase(self, x, alt):
        return self.switch(self._learning_phase, x, alt)

    def in_test_phase(self, x, alt):
        return self.switch(self._learning_phase, alt, x)

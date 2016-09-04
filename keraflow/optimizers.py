import numpy as np

from six.moves import zip

from . import backend as B
from .utils import KeraFlowError as KError


def clip_norm(g, c, n):
    if c > 0:
        g = B.switch(n >= c, g * c / n, g)
    return g


def kl_divergence(p, p_hat):
    return p_hat - p + p * B.log(p / p_hat)


class Optimizer(object):
    '''Abstract optimizer base class.
    '''
    def __init__(self, clipnorm=None, clipvalue=None):
        '''
        @param clipnorm: float >= 0. Gradients will be clipped when their L2 norm exceeds this value.
        @param clipvalue: float >= 0. Gradients will be clipped when their absolute value exceeds this value.
        '''
        if clipnorm is not None and clip_norm<=0:
            raise KError('clipnorm should be greater than 0!!')
        if clipvalue is not None and clipvalue<=0:
            raise KError("clipvalue should be greater than 0!!")
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.updates = []
        self.weights = []

    def get_state(self):
        return [B.eval(u[0]) for u in self.updates]

    def set_state(self, value_list):
        assert len(self.updates) == len(value_list)
        for u, v in zip(self.updates, value_list):
            B.set_value(u[0], v)

    def get_updates(self, loss, params):
        raise NotImplementedError

    def get_gradients(self, loss, params):
        grads = B.gradients(loss, params)
        if self.clipnorm is not None:
            norm = B.sqrt(sum([B.sum(B.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if self.clipvalue is not None:
            grads = [B.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads


class SGD(Optimizer):
    '''Stochastic gradient descent, with support for momentum, learning rate decay, and Nesterov momentum.
    '''
    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, **kwargs):
        '''
        @param lr: float >= 0. Learning rate.
        @param momentum: float >= 0. Parameter updates momentum.
        @param decay: float >= 0. Learning rate decay over each update.
        @param nesterov: boolean. Whether to apply Nesterov momentum.
        @param kwargs: see Optimizer.__init__
        '''
        super(SGD, self).__init__(**kwargs)
        self.lr = B.variable(lr)
        self.momentum = B.variable(momentum)
        self.decay = B.variable(decay)
        self.nesterov = nesterov
        self.iterations = B.variable(0.)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        lr = self.lr * (1. / (1. + self.decay * self.iterations))
        self.updates = [(self.iterations, self.iterations + 1.)]

        # momentum
        self.weights = [B.variable(np.zeros(B.eval(p).shape)) for p in params]
        for p, g, m in zip(params, grads, self.weights):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append((m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v
            self.updates.append((p, new_p))
        return self.updates


class RMSprop(Optimizer):
    '''RMSProp optimizer. This optimizer is usually a good choice for recurrent neural networks.
    '''
    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-8, **kwargs):
        '''It is recommended to leave the parameters of this optimizer at their default values (except the learning rate, which can be freely tuned).
        @param lr: float >= 0. Learning rate.
        @param rho: float >= 0.
        @param epsilon: float >= 0. Fuzz factor.
        @param kwargs: see Optimizer.__init__
        '''
        super(RMSprop, self).__init__(**kwargs)
        self.lr = B.variable(lr)
        self.rho = B.variable(rho)
        self.epsilon = epsilon

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        # accumulators
        self.weights = [B.variable(np.zeros(B.eval(p).shape)) for p in params]
        self.updates = []

        for p, g, a in zip(params, grads, self.weights):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * B.square(g)
            self.updates.append((a, new_a))
            new_p = p - self.lr * g / (B.sqrt(new_a) + self.epsilon)
            self.updates.append((p, new_p))
        return self.updates


class Adagrad(Optimizer):
    '''Adagrad optimizer.
    - Reference: [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    '''
    def __init__(self, lr=0.01, epsilon=1e-8, **kwargs):
        '''It is recommended to leave the parameters of this optimizer at their default values.
        @param lr: float >= 0. Learning rate.
        @param epsilon: float >= 0.
        @param kwargs: see Optimizer.__init__
        '''
        super(Adagrad, self).__init__(**kwargs)
        self.lr = B.variable(lr)
        self.epsilon = epsilon

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        # accumulators
        self.weights = [B.variable(np.zeros(B.eval(p).shape)) for p in params]
        self.updates = []

        for p, g, a in zip(params, grads, self.weights):
            new_a = a + B.square(g)  # update accumulator
            self.updates.append((a, new_a))
            new_p = p - self.lr * g / (B.sqrt(new_a) + self.epsilon)
            self.updates.append((p, new_p))
        return self.updates


class Adadelta(Optimizer):
    '''Adadelta optimizer.
    - Reference: [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701)
    '''
    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-8, **kwargs):
        '''It is recommended to leave the parameters of this optimizer at their default values.
        @param lr: float >= 0. Learning rate.
        @param rho: float >= 0.
        @param epsilon: float >= 0. Fuzz factor.
        @param kwargs: see Optimizer.__init__
        '''
        super(Adadelta, self).__init__(**kwargs)
        self.lr = B.variable(lr)
        self.rho = rho
        self.epsilon = epsilon

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        accumulators = [B.variable(np.zeros(B.eval(p).shape)) for p in params]
        delta_accumulators = [B.variable(np.zeros(B.eval(p).shape)) for p in params]
        self.weights = accumulators + delta_accumulators
        self.updates = []

        for p, g, a, d_a in zip(params, grads, accumulators, delta_accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * B.square(g)
            self.updates.append((a, new_a))

            # use the new accumulator and the *old* delta_accumulator
            update = g * B.sqrt(d_a + self.epsilon) / B.sqrt(new_a + self.epsilon)

            new_p = p - self.lr * update
            self.updates.append((p, new_p))

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * B.square(update)
            self.updates.append((d_a, new_d_a))
        return self.updates


class Adam(Optimizer):
    '''Adam optimizer.
    - Reference: [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

    '''
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, **kwargs):
        '''Default parameters follow those provided in the reference paper.
        @param lr: float >= 0. Learning rate.
        @param beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        @param epsilon: float >= 0. Fuzz factor.
        @param kwargs: see Optimizer.__init__
        '''
        super(Adam, self).__init__(**kwargs)
        self.lr = B.variable(lr)
        self.beta_1 = B.variable(beta_1)
        self.beta_2 = B.variable(beta_2)
        self.epsilon = epsilon
        self.iterations = B.variable(0)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations + 1)]

        t = self.iterations + 1
        lr_t = self.lr * B.sqrt(1. - B.pow(self.beta_2, t)) / (1. - B.pow(self.beta_1, t))

        ms = [B.variable(np.zeros(B.eval(p).shape)) for p in params]
        vs = [B.variable(np.zeros(B.eval(p).shape)) for p in params]
        self.weights = ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * B.square(g)
            p_t = p - lr_t * m_t / (B.sqrt(v_t) + self.epsilon)

            self.updates.append((m, m_t))
            self.updates.append((v, v_t))

            new_p = p_t
            self.updates.append((p, new_p))
        return self.updates


class Adamax(Optimizer):
    '''Adamax optimizer from Adam paper's Section 7. It is a variant of Adam based on the infinity norm.
    - Reference: [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, **kwargs):
        '''Default parameters follow those provided in the paper.
        @param lr: float >= 0. Learning rate.
        @param beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        @param epsilon: float >= 0. Fuzz factor.
        @param kwargs: see Optimizer.__init__
        '''
        super(Adamax, self).__init__(**kwargs)
        self.lr = B.variable(lr)
        self.beta_1 = B.variable(beta_1)
        self.beta_2 = B.variable(beta_2)
        self.epsilon = epsilon
        self.iterations = B.variable(0.)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations + 1)]

        t = self.iterations + 1
        lr_t = self.lr / (1. - B.pow(self.beta_1, t))

        # zero init of 1st moment
        ms = [B.variable(np.zeros(B.eval(p).shape)) for p in params]
        # zero init of exponentially weighted infinity norm
        us = [B.variable(np.zeros(B.eval(p).shape)) for p in params]
        self.weights = ms + us

        for p, g, m, u in zip(params, grads, ms, us):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            u_t = B.maximum(self.beta_2 * u, B.abs(g))
            p_t = p - lr_t * m_t / (u_t + self.epsilon)

            self.updates.append((m, m_t))
            self.updates.append((u, u_t))

            new_p = p_t
            self.updates.append((p, new_p))
        return self.updates

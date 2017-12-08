

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

class BatchNorm(object):

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return batch_norm(x,
                          decay=self.momentum,
                          updates_collections=None,
                          epsilon=self.epsilon,
                          scale=True,
                          is_training=train,
                          scope=self.name)
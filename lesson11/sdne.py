# -*- coding:utf-8 -*-

import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l1_l2

import tensorflow.contrib.slim as slim

def l_2nd(beta):
    def loss_2nd(y_true, y_pred):
        b_ = tf.ones_like(y_true)
        not_zero = tf.not_equal(y_true, 0)
        b_ = tf.multiply(tf.to_int32(tf.logical_not(not_zero)), b_) + tf.multiply(tf.to_int32(not_zero), beta)
        x = tf.square(tf.subtract(y_true, y_pred) * b_)
        t = tf.reduce_mean(x, axis=-1)
        return t
    return loss_2nd

def l_1st(alpha):
    pass

def create_model(node_size, hidden_size, l1=1e-5, l2=1e-4):
    A = tf.placeholder(dtype=tf.float32, shape=(node_size,))
    L = tf.placeholder(dtype=tf.float32, shape=(None,))

    fc = A
    for i in range(len(hidden_size)):
        if i == len(hidden_size) - 1:
            fc = slim.fully_connected(fc, hidden_size[i], scope=f"1st")
        else:
            fc = slim.fully_connected(fc, hidden_size[i], scope=f"fc_{i}")

    Y = fc
    for i in reversed(range(len(hidden_size) - 1)):
        fc = slim.fully_connected(fc, hidden_size[i], scope=f"fc_reversed_{i}")

    A_ = slim.fully_connected(fc, node_size, scope="2nd")




class SDNE(object):
    def __init__(self, nx_graph, hidden_size=[32, 16], alpha=1e-6, beta=5., nu1=1e-5, nu2=1e-4):
        self.G = nx_graph
        self.node_size = self.G.number_of_nodes()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2


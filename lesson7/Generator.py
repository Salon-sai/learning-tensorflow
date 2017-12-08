# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import slim

def generator(noise):
    pass


def build_model(input):
    with slim.arg_scope([slim.conv2d_transpose]):

        with tf.variable_scope("generator") as scope:
            pass

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

        if with_w:
            return tf.add(tf.matmul(input_, matrix), bias), matrix, bias
        else:
            return tf.add(tf.matmul(input_, matrix), bias)
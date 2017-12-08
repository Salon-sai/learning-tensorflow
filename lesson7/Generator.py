# -*- coding: utf-8 -*-

import tensorflow as tf

from lesson7.utils import BatchNorm
from tensorflow.contrib import slim

def generator(noise, image_size=64):
    pass


def build_model(input, image_size=64):
    with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5, 5], stride=2,
                        activation_fn=None):
        net = linear(input, 2 * image_size * image_size, 'generator/linear_1') # output_size=2^13
        net = tf.reshape(net, [-1, image_size // 16, image_size // 16, 512], name='generator/reshape_2')
        net = BatchNorm(net, name="batch_norm_3")
        net = tf.nn.relu(net)

        net = slim.conv2d_transpose(inputs=net, num_outputs=256, padding="SAME", name="generator/deconv_4")
        net = BatchNorm(net, name="batch_norm_5")
        net = tf.nn.relu(net)

        net = slim.conv2d_transpose(inputs=net, num_outputs=128, padding="SAME", name="generator/deconv_6")
        net = BatchNorm(net, name="batch_norm_7")
        net = tf.nn.relu(net)

        net = slim.conv2d_transpose(inputs=net, num_outputs=64, padding="SAME", name="generator/deconv_8")
        net = BatchNorm(net, name="batch_norm_9")
        net = tf.nn.relu(net)

        net = slim.conv2d_transpose(inputs=net, num_outputs=3, padding="SAME", name="generator/deconv_10")
        net = tf.nn.tanh(net)
    return net

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
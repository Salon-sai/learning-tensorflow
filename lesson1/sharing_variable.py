# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

input_images = tf.placeholder(tf.float32, shape=(1, 32, 32, 1))

# 定义了一层卷积神经网络
def conv_relu(input, kernel_shape, bias_shape):
    # 创建名为weights的变量
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    # 创建名为biases的变量
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))

    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')

    return tf.nn.relu(conv + biases)

def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # 在名为conv1的variable scope下调用一层神经网络，对应的参数名为
        # "conv1/weights", "conv1/biases"
        relu1 = conv_relu(input_images, [3, 3, 1, 1], [1])
    with tf.variable_scope("conv2"):
        # 在名为conv2的variable scope下调用一层神经网络，对应的参数名为
        # "conv2/weights", "conv2/biases"
        return conv_relu(relu1, [3, 3, 1, 1], [1])

with tf.variable_scope("image_filter") as scope:
    result1 = my_image_filter(input_images)
    # 重用变量
    scope.reuse_variables()
    result2 = my_image_filter(input_images)

init = tf.global_variables_initializer();

with tf.Session() as sess:
    sess.run(init)
    image = np.random.rand(1, 32, 32, 1)
    result1 = sess.run(result1, feed_dict={input_images: image})
    result2 = sess.run(result2, feed_dict={input_images: image})

    print(result2.all() == result1.all())


# -*- coding: utf-8 -*-

import tensorflow as tf

# Variable定义了weights作为一个变量，传入的参数是该变量的值和名字
# random_normal的是产生高斯分布的随机变量，而给定参数分别是变量的维数，期望值，方差
weights = tf.Variable(tf.random_normal([2, 2, 3], mean=0, stddev=0.35), name="weights")

biases = tf.Variable(tf.zeros([200]), name="biases")

w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")

# 我们定义了变量后，需要将所有变量初始化
# init_op = tf.global_variables_initializer()
# 我们只将部分变量初始化
init_op = tf.variables_initializer({weights, w_twice})

with tf.Session() as sess:
    sess.run(init_op)
    print("\n")
    print(sess.run(weights))
    print("\n")
    print(sess.run(w_twice))



# -*- coding: utf-8 -*-

import tensorflow as tf
import time

# 训练数据一共有128个
x_input_data = tf.random_normal([128, 10], mean=0, stddev=1)

with tf.variable_scope("queue"):
    # enqueue 5 batches
    q = tf.FIFOQueue(capacity=5, dtypes=tf.float32)

    x_input_data = tf.Print(x_input_data, data=[x_input_data], message="Raw inputs data generated:", summarize=6)


    enqueue_op = q.enqueue(x_input_data)

    numberOfThread = 1

    qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThread)
    tf.train.add_queue_runner(qr)

    # 训练样本
    input = q.dequeue()
    input = tf.Print(input, data=[q.size(), input], message="queue size ,input : ")
    # 样本对应的标签
    y_true = tf.cast(tf.reduce_sum(input, axis=1, keep_dims=True) > 0, tf.int32)

with tf.variable_scope('FullyConnected'):
    w = tf.get_variable('w', shape=[10, 1024], initializer=tf.random_normal_initializer(stddev=1e-1))
    b = tf.get_variable('b', shape=[1024], initializer=tf.constant_initializer(0.1))

    z = tf.matmul(input, w) + b

    y = tf.nn.relu(z)

    w2 = tf.get_variable('w1', shape=[1024, 1], initializer=tf.random_normal_initializer(stddev=1e-1))
    b2 = tf.get_variable('b1', shape=[1], initializer=tf.constant_initializer(0.1))

    z = tf.matmul(y, w2) + b2

with tf.variable_scope('Loss'):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(None, tf.cast(y_true, tf.float32), z)
    loss_op = tf.reduce_mean(loss)

with tf.variable_scope("Accuracy"):
    y_pred = tf.cast(z > 0, tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))
    accuracy = tf.Print(accuracy, data=[accuracy], message="accuary:")

adam = tf.train.AdamOptimizer(1e-2)
train_op = adam.minimize(loss_op, name="train_op")

startTime = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    sess.run(accuracy)

    for i in range(128):
        _, loss = sess.run([train_op, loss_op])

        # We regularly check the loss
        print('iter:%d - loss:%f' % (i, loss))

    sess.run(accuracy)

    coord.request_stop()
    coord.join(threads)

print("Time taken: %f" % (time.time() - startTime))


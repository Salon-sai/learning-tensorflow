# -*- coding: utf-8 -*-

import tensorflow as tf

# 定义输入数据，一共有9个
x_input_data = tf.random_normal([6], mean=-1, stddev=4)

# 定义一个容量为2的队列
q = tf.FIFOQueue(capacity=3, dtypes=tf.float32)

x_input_data = tf.Print(x_input_data, data=[x_input_data], message="Raw inputs data generated:", summarize=6)

# 注入多个值进入队列
enqueue_op = q.enqueue_many(x_input_data)

numbreOfThreads = 1
# 定义queue runner
qr = tf.train.QueueRunner(q, [enqueue_op] * numbreOfThreads)

# 将queue runner集合中
tf.train.add_queue_runner(qr)

input = q.dequeue()
input = tf.Print(input, data=[q.size(), input], message="Nb element left, input:")

# 假设开始训练
y = input + 1

with tf.Session() as sess:
    # 创建协调者
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(coord=coord)

    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)


    coord.request_stop()

    coord.join(threads)

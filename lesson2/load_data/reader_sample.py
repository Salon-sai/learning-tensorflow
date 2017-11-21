# -*- coding: utf-8 -*-

import tensorflow as tf

# 生成一个先入先出队列和一个QueueRunner
filenames = ['A.csv', 'B.csv', 'C.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

# 可以理解为一个dequeue操作
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])

with tf.Session() as session:
    coord = tf.train.Coordinator()
    # 启动QueueRunner，此时文件名队列已经进队
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        ex, lab = session.run([example, label])
        print(ex, lab)
    coord.request_stop()
    coord.join(threads)
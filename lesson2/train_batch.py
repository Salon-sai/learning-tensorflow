# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import data_flow_ops

tensor_list = np.reshape(np.arange(20), (-1, 4)).tolist()
tensor_list1 = [[[1, 2, 3], [3, 4, 5], [1, 2, 3], [3, 4, 5]], [[2, 22, 5], [3, 2, 6], [8, 22, 5], [9, 2, 6]]]
tensor_list2 = np.reshape(np.arange(30), (-1, 3))

list_placehodler = tf.placeholder(tf.int64, shape=(None, 3), name="list_placehodler")
input_queue = data_flow_ops.FIFOQueue(capacity=10000, dtypes=[tf.int64], shapes=[(3, )], shared_name=None, name=None)

enqueue_op = input_queue.enqueue_many([list_placehodler])

lists = []
# for _ in range(2):
dequeue_list = input_queue.dequeue()
lists.append([dequeue_list])

list_batch = tf.train.batch_join(lists, batch_size=2, shapes=[()], enqueue_many=True, capacity=2)

print(list_batch)

with tf.Session() as session:
    # batch与batch_join的主要区别在于：
    # batch是多线程读取文件，而batch_join是多线程读取文件。因此batch的读取不会像batch_join
    # 出现样本打乱或者样本重复（因为多线程会存在并发的问题）

    # batch:
    # enqueue_many=False：把tensor_list中的每个tensor看作一个样本，要生成一个batch，它就复制该样本batch_size条
    # 通常tensor_list存放做一个文件内容，每行数据重复读取batch_size次来生成batch
    # enqueue_many=True：把tensor_list看作一个tensor看作一个样本batch。
    # 而batch_size可以截取样本数目或者补充数据（补充在该batch的样本）来维持每个batch_size的大小
    
    session.run(enqueue_op, feed_dict={list_placehodler: tensor_list2})

    x1 = tf.train.batch(tensor_list, batch_size=5, enqueue_many=False)
    x2 = tf.train.batch(tensor_list, batch_size=5, enqueue_many=True)

    # batch_join:
    # enqueue_many=False: 与batch一样，它把每个tensor看作一个样本，但他并没有复制多条样本，
    # 而是把多个样本数据分开在每个batch_list中。batch的shape大概是len(tensor_list), batch_size
    y1 = tf.train.batch_join(tensor_list, batch_size=6, enqueue_many=False)
    y2 = tf.train.batch_join(tensor_list1, batch_size=2, enqueue_many=True)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    print("x1 batch: " + "-" * 10)

    x1_value = session.run(x1)
    print(x1_value)

    print("x2 batch: " + "-" * 10)

    x2_value = session.run(x2)
    print(x2_value)

    print("y1 batch: " + "-" * 10)

    y1_value = session.run(y1)
    print(y1_value)

    print("y2 batch: " + "-" * 10)

    for _ in range(10):
        y2_value = session.run(y2)
        print(y2_value)

    print("join queueu batch" + "-" * 10)

    for _ in range(13):
        list_batch_output = session.run(list_batch)
        print(list_batch_output)
    # session.run(enqueue_op, feed_dict={list_placehodler: tensor_list2})

    coord.request_stop()
    coord.join(threads)

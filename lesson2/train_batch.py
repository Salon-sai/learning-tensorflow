# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

tensor_list = np.reshape(np.arange(20), (-1, 4)).tolist()
tensor_list1 = np.reshape(np.arange(20), (-1, 1, 4)).tolist()
# tensor_list

with tf.Session() as session:
    # batch与batch_join的主要区别在于：
    # batch是多线程读取文件，而batch_join是多线程读取文件。因此batch的读取不会像batch_join
    # 出现样本打乱或者样本重复（因为多线程会存在并发的问题）

    # batch:
    # enqueue_many=False：把tensor_list中的每个tensor看作一个样本，要生成一个batch，它就复制该样本batch_size条
    # 通常tensor_list存放做一个文件内容，每行数据重复读取batch_size次来生成batch
    # enqueue_many=True：把tensor_list看作一个tensor看作一个样本batch。
    # 而batch_size可以截取样本数目或者补充数据（补充在该batch的样本）来维持每个batch_size的大小
    x1 = tf.train.batch(tensor_list, batch_size=5, enqueue_many=False)
    x2 = tf.train.batch(tensor_list, batch_size=5, enqueue_many=True)

    # batch_join:
    # enqueue_many=False: 与batch一样，它把每个tensor看作一个样本，但他并没有复制多条样本，
    # 而是把多个样本数据分开在每个batch_list中。batch的shape大概是len(tensor_list), batch_size
    y1 = tf.train.batch_join(tensor_list, batch_size=6, enqueue_many=False)
    y2 = tf.train.batch_join(tensor_list1, batch_size=4, enqueue_many=True)

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

    y2_value = session.run(y2)
    print(y2_value)

    coord.request_stop()
    coord.join(threads)

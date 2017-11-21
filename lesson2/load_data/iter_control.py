# -*- coding: utf-8 -*-

import tensorflow as tf

# 生成一个先入先出队列和一个QueueRunner
filenames = ['D.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

reader = tf.TextLineReader()
key, value = reader.read(queue=filename_queue)

example_list = [tf.decode_csv(value, record_defaults=[['null'], ['null']]) for _ in range(2)]
# example_list = [tf.decode_csv(value, record_defaults=[['null'], ['null']])]

example_batch, label_batch = tf.train.batch_join(
    tensors_list=example_list,
    batch_size=5
)

init_local_op = tf.local_variables_initializer()

with tf.Session() as session:
    session.run(init_local_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():
            examples, labels = session.run([example_batch, label_batch])
            print(labels, examples)
    except tf.errors.OutOfRangeError:
        print('Epochs Complete !')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    coord.request_stop()



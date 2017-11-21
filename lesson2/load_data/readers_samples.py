# -*- coding: utf-8 -*-

import tensorflow as tf

filenames = ['A.csv', 'B.csv', 'C.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

example_list = [tf.decode_csv(value, record_defaults=[['null'], ['null']]) for _ in range(2)]

example_batch, label_batch = tf.train.batch_join(example_list, batch_size=3)

with tf.Session() as session:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10):
        examples, labels = session.run([example_batch, label_batch], feed_dict=None)
        print(labels, examples)

    coord.request_stop()
    coord.join(threads)

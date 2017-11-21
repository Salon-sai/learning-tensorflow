# -*- coding: utf-8 -*-

import tensorflow as tf

filenames = ['A.csv', 'B.csv', 'C.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])

example_batch, label_batch = tf.train.batch(
    tensors=[example, label],
    batch_size=5,
    capacity=32,
    enqueue_many=False)

with tf.Session() as session:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        examples, labels = session.run(example_batch, label_batch)
        print(labels, examples)

    coord.request_stop()
    coord.join(threads=threads)

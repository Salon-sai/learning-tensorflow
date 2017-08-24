# -*-coding:utf-8-*-#
import tensorflow as tf

import reader
from BatchGenerator import BatchGenerator
from GeneratePopetryModel import GeneratePopetryModel

dictionary, poetry_vectors, _ = reader.build_dataset()

empty_key = dictionary.get(' ')

batch_size =64

batch_generator = BatchGenerator(poetry_vectors, batch_size, empty_key)

# x_data, y_data = batch_generator.next()

input_size = output_size = len(dictionary) + 1

train_data = tf.placeholder(tf.int32, [batch_size, None])
train_label = tf.placeholder(tf.int32, [batch_size, None])

model = GeneratePopetryModel(X=train_data, batch_size=batch_size, input_size=input_size, output_size=output_size)

logists, last_state, _ = model.results()
targets = tf.reshape(train_label, [-1])
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logists], [targets], [tf.ones_like(targets, dtype=tf.float32)], len(dictionary))
cost = tf.reduce_mean(loss)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.01, global_step, batch_generator._batch_num, 0.97, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients, v = zip(*optimizer.compute_gradients(cost))
gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())
    print("training...")
    for epoch in range(50):
        for batch in range(batch_generator._batch_num):
            x_data, y_data = batch_generator.next()
            _, _, c, lr, gs = session.run(
                [optimizer, last_state, cost, learning_rate, global_step],
                feed_dict={train_data: x_data, train_label: y_data})
            print(epoch, batch, c, lr, gs)

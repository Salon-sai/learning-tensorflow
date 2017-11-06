import tensorflow as tf
import dnn_d_vector as d_v
import data_process as dp
import numpy as np
import os

BATCH_SIZE = 64
epoch = 20
model_dir = './model/'

train_data, train_label = dp.get_train_data()
batch_num = train_data.shape[0] // BATCH_SIZE + train_data.shape[0] % BATCH_SIZE
label_num = train_label.shape[1]
generator = dp.generate_batch(BATCH_SIZE, train_data, train_label)

global_step = tf.Variable(0, trainable=False)

X = tf.placeholder(tf.float32, [None, 40])
Y = tf.placeholder(tf.float32, [None, label_num])

d_vector = d_v.d_vector_model(X)

with tf.variable_scope("predict") as predict_layer:
    predict = d_v.nn_layer(d_vector, [256, label_num], [label_num], dropout=False, need_norm=False)
    d_v.variable_summaries(predict, 'layer4_predict')

with tf.name_scope('accuracy'):
    correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
    valid_accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    tf.summary.scalar('train_accuracy', valid_accuracy)

loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))

learning_rate = tf.train.exponential_decay(1e-3, global_step, 50000, 0.1, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function, global_step=global_step)

ckpt = tf.train.get_checkpoint_state('./model')
saver = tf.train.Saver()
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/', session.graph)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    if ckpt != None:
        print(ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
        data, label = next(generator)
        accuracy = session.run(valid_accuracy, feed_dict={X: data, Y: label, d_v.train_phase: False})
        print(accuracy)

    else:
        print("no model")
        # exit()
    
        i = 0
        while i < epoch:

            for batch in range(batch_num):
                data, label = next(generator)
                # print(data)
                # print(label)
                _, l, lr, gs, summary, accuracy = session.run([optimizer, loss_function, learning_rate, global_step, \
                    merged, valid_accuracy], feed_dict={X: data, Y: label, d_v.train_phase: True})
                print(l, lr, gs)

                if gs % 1000 == 0:
                    writer.add_summary(summary, gs)

                if gs % 5000 == 0:
                    saver.save(session, model_dir + 'd_vector.module', global_step=gs)
                    # accuracy = session.run(valid_accuracy, feed_dict={X: data, Y: label, d_v.train_phase: False})
                    print("\n %1.5f \n" % accuracy)
            i += 1
            saver.save(session, model_dir + 'd_vector.module_%d' % i)

        # gs = 0
        # while gs < 5000:
        #     data, label = next(generator)
        #     # print(data)
        #     # print(label)
        #     _, l, lr, gs, summary = session.run([optimizer, loss_function, learning_rate, global_step, merged], feed_dict={X: data, Y: label, d_v.train_phase: True})
        #     print(l, lr, gs)
        #     writer.add_summary(summary, gs)

        #     if gs % 5000 == 0:
        #         saver.save(session, model_dir + 'd_vector.module', global_step=gs)

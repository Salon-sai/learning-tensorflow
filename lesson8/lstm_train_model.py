import tensorflow as tf
import lstm_d_vector as d_v
import data_process as dp
import numpy as np
import os

BATCH_SIZE = 1
FEATURE_SIZE = 40
D_VECTOR_DIM = 256

train_data, train_label = dp.load_data_with_wavfile(dp.TRAIN_PATH)
print(train_label)
# batch_num = train_data.shape[0] // BATCH_SIZE + train_data.shape[0] % BATCH_SIZE
# generator = dp.generate_batch(BATCH_SIZE, train_data, train_label)

epoch = 20
label_num = train_label.shape[1]
model_dir = './lstm-model/'


def train_rnn_d_vector():
    X = tf.placeholder(tf.float32, [BATCH_SIZE, None, FEATURE_SIZE])
    Y = tf.placeholder(tf.float32, [BATCH_SIZE, label_num])

    d_vectors = d_v.RNN_d_vector(X, D_VECTOR_DIM)

    with tf.variable_scope("predict") as predict_scope:
        weight = tf.get_variable('weight', [D_VECTOR_DIM, label_num], initializer=tf.random_normal_initializer())
        d_v.variable_summaries(weight, "weight")
        bias = tf.get_variable('bias', [label_num], initializer=tf.random_normal_initializer())
        d_v.variable_summaries(bias, "bias")
        predict = tf.matmul(d_vectors, weight) + bias
        d_v.variable_summaries(predict, "predict")
        cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
        tf.summary.scalar('loss_func', cost_func)
        
    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
        valid_accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        tf.summary.scalar('train_accuracy', valid_accuracy)

    optimizer = tf.train.AdamOptimizer().minimize(cost_func)
    
    ckpt = tf.train.get_checkpoint_state('./lstm-model')
    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('lstm-logs/', session.graph)

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        if ckpt != None:
            print(ckpt.model_checkpoint_path)
            saver.restore(session, ckpt.model_checkpoint_path)
            accuracy = session.run(valid_accuracy, feed_dict={X: data, Y: label})
            print(accuracy)
        else:
            i = 0
            while i < epoch:
                for wav_data, label in zip(train_data, train_label):
                    _, l, accuracy = session.run([optimizer, cost_func, valid_accuracy], feed_dict={X: np.array([wav_data]), Y: np.array([label])})
                    print("loss: ", l)

                    if gs % 1000 == 0:
                        writer.add_summary(summary, gs)

                    if gs % 5000 == 0:
                        saver.save(session, model_dir + 'd_vector.module', global_step=gs)
                        # accuracy = session.run(valid_accuracy, feed_dict={X: data, Y: label, d_v.train_phase: False})
                        print("\n %1.5f \n" % accuracy)
                i += 1
                saver.save(session, model_dir + 'd_vector.module_%d' % i)

train_rnn_d_vector()


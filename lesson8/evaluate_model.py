import tensorflow as tf
import d_vector as d_v
import data_process as dp
import numpy as np

test_data, _ = dp.get_test_data()

ckpt = tf.train.get_checkpoint_state('./model')
X = tf.placeholder(tf.float32, [None, 40])

ds = d_v.d_vector_model(X)
# d_vector = tf.reduce_mean(d_v.d_vector_model(X), 0)

saver = tf.train.Saver()
with tf.Session() as session:
    if ckpt != None:
        print(ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("no model")
        exit()

    representation_data = [data[0] for data in test_data]
    # print(len(representation_data))
    representation_vector = []
    for data in representation_data:
        # print(data)
        vector = session.run([ds], feed_dict={X: data})
        # print(len(vector))
        print(vector)

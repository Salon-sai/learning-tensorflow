# -*- coding: utf-8 -*-

import tensorflow as tf
import os

v1 = tf.Variable(tf.random_normal([2, 2], mean=1, stddev=0.2), name="v1")
v2 = tf.Variable(tf.random_normal([3, 3], mean=3, stddev=0.5), name="v2")

init_op = tf.global_variables_initializer()

# 创建saver对象
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(v1))
    print("\n")
    print(sess.run(v2))

    if not os.path.exists('./tmp'):
        print("create the directory: ./tmp")
        os.mkdir("./tmp")

    # 保存好模型参数
    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)


with tf.Session() as sess1:
    # 恢复模型参数到sess1当中
    saver.restore(sess1, "./tmp/model.ckpt")
    print("Model restored")
    print(sess1.run(v1))

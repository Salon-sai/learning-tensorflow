# -*-coding:utf-8-*-#
import numpy as np
import tensorflow as tf

import reader
from GeneratePoetryModel import GeneratePoetryModel

dictionary, _, reversed_dictionary = reader.build_dataset()

def to_word(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    return reversed_dictionary[sample]

input_data = tf.placeholder(tf.int32, [1, None])
input_size = output_size = len(reversed_dictionary) + 1
model = GeneratePoetryModel(X=input_data, batch_size=1, input_size=input_size, output_size=output_size)
_, last_state, probs, initial_state = model.results()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    print("generate...")
    saver.restore(session, './model/poetry.module-49')
    x = np.array([list(map(dictionary.get, '['))])
    state_ = session.run(initial_state)
    word = poem = '['
    while word != ']':
        probs_, state_ = session.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
        word = to_word(probs_)
        poem += word
        x = np.zeros((1, 1))
        x[0, 0] = dictionary[word]
    print(poem)

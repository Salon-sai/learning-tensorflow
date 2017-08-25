# -*-coding:utf-8-*-#
import numpy as np
import tensorflow as tf

import reader
from GeneratePoetryModel import GeneratePoetryModel

dictionary, _, reversed_dictionary = reader.build_dataset()

def to_word(weights):
    """
    通过传入的权重，计算向量的概率分布并通过随机采样获得最接近的词语，
    类似遗传算法的选择步骤。（个人认为不够严谨）
    """
    t = np.cumsum(weights)
    s = np.sum(weights)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    return reversed_dictionary[sample]

# 定义输入的只有一个字词，然后根据上一个字词推测下一个词的位置
input_data = tf.placeholder(tf.int32, [1, None])
# 输入和输出的尺寸为1
input_size = output_size = len(reversed_dictionary) + 1
# 定义模型
model = GeneratePoetryModel(X=input_data, batch_size=1, input_size=input_size, output_size=output_size)
# 获取模型的输出参数
_, last_state, probs, initial_state = model.results()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    print("generate...")
    saver.restore(session, './model/poetry.module-140')
    # 起始字符是'['，
    x = np.array([list(map(dictionary.get, '['))])
    # 运行初始0状态
    state_ = session.run(initial_state)
    word = poem = '['
    # 结束字符是']'
    while word != ']':
        # 使用上一级的state和output作为输入
        probs_, state_ = session.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
        word = to_word(probs_)
        poem += word
        # 获取词语的id
        x = np.zeros((1, 1))
        x[0, 0] = dictionary[word]
    print(poem)

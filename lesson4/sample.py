# -*-coding:utf-8-*-#

import random
import numpy as np
from config import config

def sample_distribution(distribution):
    # 随机概率分布采样
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1

def sample(prediction):
    # 随机采样生成one-hot向量
    p = np.zeros(shape=[1, config.vocabulary_size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p

def random_distribution():
    # 生成随机概率向量,向量大小为1*27
    b = np.random.uniform(0.0, 1.0, size=[1, config.vocabulary_size])
    return b / np.sum(b, 1)[:, None]

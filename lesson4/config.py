# -*-coding:utf-8-*-#
import string

class ModelConfig(object):
    def __init__(self):
        self.num_unrollings = 10 # 每条数据的字符串长度
        self.batch_size = 64 # 每一批数据的个数
        self.vocabulary_size = len(string.ascii_lowercase) + 1 # 定义出现字符串的个数(一共有26个英文字母和一个空格)
        self.summary_frequency = 100 # 生成样本的频率
        self.num_steps = 7001 # 训练步数
        self.num_nodes = 64 # 隐含层个数

config = ModelConfig()


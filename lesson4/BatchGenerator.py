# -*-coding:utf-8-*-#
import numpy as np
from handleData import char2id
from config import config

class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        # 每个串之间的间距
        segment = self._text_size // self._batch_size
        # 记录每个串当前的位置
        self._cursor =[ offset * segment for offset in range(self._batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """
        从当前数据的游标位置生成单一批数据，一个batch的大小为(batch, 27)
        """
        batch = np.zeros(shape=(self._batch_size, config.vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            # 生成one-hot向量
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        # 因为这里加入了上一批数据的最后一个字符，所以当前这批
        # 数据每串长度为num_unrollings + 1
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches

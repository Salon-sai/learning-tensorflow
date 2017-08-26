# -*-coding:utf-8-*-#
import numpy as np
import reader

class BatchGenerator(object):
    """docstring for BatchGenerator"""
    def __init__(self, data, batch_size, empty_key):
        self._batch_size = batch_size
        self._offset = 0
        self._batch = []
        self._data_size = len(data)
        self._batch_num = self._data_size // self._batch_size
        self._data = data
        self._generate_batch(empty_key)

    def _generate_batch(self, empty_key):
        for index in range(self._batch_num):
            start = index * self._batch_size
            end = start + self._batch_size
            # 当前batch中诗词的最大长度
            length = max(map(len, self._data[start: end]))
            # 创建batch数据，假如有诗词没有达到最大长度使用空格作为补充
            batch_data = np.full((self._batch_size, length), empty_key, np.int32)
            for row in range(self._batch_size):
                poetry = self._data[start + row]
                batch_data[row, :len(poetry)] = poetry
            self._batch.append(batch_data)

    def next(self):
        x_data = self._batch[self._offset]
        y_data = np.copy(x_data)
        y_data[:, : -1] = x_data[:, 1: ]
        self._offset = (self._offset + 1) % self._batch_num
        return x_data, y_data

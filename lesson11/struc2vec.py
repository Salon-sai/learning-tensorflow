# -*- coding:utf-8 -*-
import math
import os
import shutil
from collections import ChainMap, deque

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from gensim.models import Word2Vec
from joblib import Parallel, delayed

class Struc2Vec():
    def __init__(self, nx_graph, walk_length=10, num_walks=100, workers=1, verbose=0, stay_prob=0.3, opt1_reduce_len=True, opt2_reduce_sim_calc=True, opt3_num_layers=None, temp_path='./temp_struc2vec/', reuse=False):
        self.graph = nx_graph

        self.opt1_reduce_len = opt1_reduce_len
        self.opt2_reduce_sim_calc = opt2_reduce_sim_calc
        self.opt3_num_layers = opt3_num_layers

        self.reuse = reuse
        self.temp_path = temp_path

        if not os.path.exists(self.temp_path):
            os.mkdir(self.temp_path)
        if not reuse:
            shutil.rmtree(self.temp_path)
            os.mkdir(self.temp_path)
        
        self.create_context_graph(self.opt3_num_layers, workers, verbose)

    def create_context_graph(self, max_num_layers, workers=1, verbose=0):
        pass

    def _compute_ordered_degreelist(self, max_num_layers):
        degree_list = []
        vertices = self.graph.nodes()
        for v in vertices:
            # 根据不同的node建立不同的树
            degree_list[int(v)] = self._get_order_degreelist_node(v, max_num_layers)
        return degree_list

    # 顶点对距离计算
    # 用一个循环去计算每个顶点对应的有序度序列。
    def _get_order_degreelist_node(self, root, max_num_layers=None):
        if max_num_layers is None:
            max_num_layers = float('inf')
        
        ordered_degree_sequence_dict = {}
        visited = [False] * len(self.graph.nodes())
        queue = deque()
        level = 0
        queue.append(root)
        visited[root] = True

        while (len(queue) > 0 and level <= max_num_layers):
            count = len(queue)
            if self.opt1_reduce_len:
                degree_list = {}
            else:
                degree_list = []

            while (count > 0):
                node = queue.popleft()
                # 获取邻居个数（度）
                degree = len(self.graph[node])
                if self.opt1_reduce_len:
                    # 统计当前层下的同样度的节点个数
                    degree_list[degee] = degree_list.get(degree, 0) + 1
                else:
                    degree_list.append(degree)
                
                # 将结论邻居放入队列当中
                for i in self.graph[node]:
                    if not visited[int(i)]:
                        visited[int(i)] = True
                        queue.append(int(i))
                count -= 1

            if self.opt1_reduce_len:
                order_degree_list = [(degree, freq) for degree, freq in degree_list.items()]
                order_degree_list.sort(key=lambda x: x[0])
            else:
                order_degree_list = sorted(degree_list)
            ordered_degree_sequence_dict[level] = order_degree_list
            level += 1
        
        return ordered_degree_sequence_dict

    def _compute_structural_distance(self, max_num_layers, workers=1, verbose=0):
        if os.path.exists(self.temp_path + 'structural_dist.pkl'):
            structural_dist = pd.read_pickle(self.temp_path+'structural_dist.pkl')
        else:
            if self.opt1_reduce_len:
                dist_func = cost_max
            else:
                dist_func = cost

            if os.path.exists(self.temp_path + 'degreelist.pkl'):
                degreeList = pd.read_pickle(self.temp_path + 'degreelist.pkl')
            else:
                degreeList = self._compute_ordered_degreelist(max_num_layers)
                pd.to_pickle(degreeList, self.temp_path + 'degreelist.pkl')
            
            if self.opt2_reduce_sim_calc:
                degree = self._create_vectors()

    def _create_vectors(self):
        degrees = {}
        degrees_sorted = set()
        G = self.graph
        # for v in self.

def cost(a, b):
    ep = 0.5
    m = max(a, b) + ep
    mi = min(a, b) + ep
    return ((m / mi) - 1)

def cost_min(a, b):
    ep = 0.5
    m = max(a[0], b[0]) + ep
    mi = min(a[0], b[0]) + ep
    return ((m / mi) - 1) * min(a[1], b[1])

def cost_max(a, b):
    ep = 0.5
    m = max(a[0], b[0]) + ep
    mi = min(a[0], b[0]) + ep
    return ((m / mi) - 1) * max(a[1], b[1])




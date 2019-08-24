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
                degree_list = pd.read_pickle(self.temp_path + 'degreelist.pkl')
            else:
                degree_list = self._compute_ordered_degreelist(max_num_layers)
                pd.to_pickle(degree_list, self.temp_path + 'degreelist.pkl')
            
            if self.opt2_reduce_sim_calc:
                degrees = self._create_vectors()
                degree_list_selected = {}
                vertices = {}
                n_nodes = len(self.graph.nodes())
                for v in self.graph.nodes():
                    # 获取相邻的节点（度数越相近越相邻）
                    nbs = get_vertices(v, len(self.graph[v]), degrees, n_nodes)
                    vertices[v] = nbs
                    degree_list_selected[v] = degree_list[v]
                    for n in nbs:
                        degree_list_selected[n] = degree_list[n]
            else:
                vertices = {}
                for v in degree_list:
                    vertices[v] = [vd for vd in degree_list.keys() if vd > v]

            results = Parallel(n_jobs=workers, verbose=verbose,)(delayed(compute_dtw_dist)(part_list, degree_list, dist_func) for part_list in partition_dict(vertices, workers))
            dtw_dist = dict((ChainMap(*results)))
            structural_dist = convert_dtw_struc_dist(dtw_dist)
            pd.to_pickle(structural_dist, self.temp_path + 'structural_dist.pkl')
        return structural_dist

    def _create_vectors(self):
        degrees = {} # key: graph中的度数，value: 度数的节点
        degrees_sorted = set() # 存放graph当中的度数
        for v in self.graph.nodes():
            degree = len(self.graph[v])
            degrees_sorted.add(degree)
            if (degree not in degrees):
                degrees[degree] = {}
                degrees[degree]['vertices'] = []
            degrees[degree]['vertices'].append(v)
        degrees_sorted = np.array(list(degrees_sorted), dtype='int')
        degrees_sorted = np.sort(degrees_sorted)

        l = len(degrees_sorted)
        for index, degree in enumerate(degrees_sorted):
            if (index > 0):
                # 记录比该度数小的前一个度数值
                degrees[degree]['before'] = degrees_sorted[index - 1]
            if (index < (l - 1)):
                # 记录比该度数大的后一个度数值
                degrees[degree]['after'] = degrees_sorted[index + 1]
        return degrees

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

def get_vertices(v, degree_v, degrees, n_nodes):
    # 通过传入的度数，找出对应的节点
    a_vertices_selected = 2 * math.log(n_nodes, 2) # 限制选择节点的个数
    vertices = []
    try:
        c_v = 0
        for v2 in degrees[degree_v]['vertices']:
            if (v != v2):
                vertices.append(v2)
                c_v += 1
                if (c_v > a_vertices_selected):
                    raise StopIteration

        # 判断是否为最小的度
        if ('before' not in degrees[degree_v]):
            degree_b = -1
        else:
            degree_b = degrees[degree_v]['before']
        
        # 判断是否为最大的度
        if ('after' not in degrees[degree_v]):
            degree_a = -1
        else:
            degree_a = degrees[degree_v]['after']

        # graph中只有一个度
        if (degree_b == -1 and degree_a == -1):
            raise StopIteration
        degree_now = verify_degrees(degrees, degree_v, degree_a, degree_b)

        while True:
            for v2 in degrees[degree_now]['vertices']:
                if (v != v2):
                    vertices.append(v2)
                    c_v += 1
                    if (c_v > a_vertices_selected):
                        raise StopIteration

            if (degree_now == degree_b):
                if ('before' not in degrees[degree_b]):
                    degree_b = -1
                else:
                    degree_b = degrees[degree_b]['before']
            else:
                if ('after' not in degrees[degree_a]):
                    degree_a = -1
                else:
                    degree_a = degrees[degree_a]['after']
            # TODO：是否需要删除这个判断，因为之前已经判断图中只有一个度
            if (degree_b == -1 and degree_a == -1):
                raise StopIteration
            degree_now = verify_degrees(degrees, degree_v, degree_a, degree_b)
    except StopIteration:
        return list(vertices)
    return list(vertices)

def verify_degrees(degrees, degree_v_root, degree_a, degree_b):
    if (degree_b == -1):
        degree_now = degree_a
    elif (degree_a == -1):
        degree_now = degree_b
    # 选择距离较近的root的那个度
    elif (abs(degree_b, degree_v_root) < abs(degree_a - degree_v_root)):
        degree_now = degree_b
    else:
        degree_now = degree_a
    return degree_now

def compute_dtw_dist(part_list, degree_list, dist_func):
    dtw_dict = {}
    for v1, nbs in part_list:
        lists_v1 = degree_list[v1] # orderd degree list of v1
        for v2 in nbs:
            lists_v2 = degree_list[v2] # orderd degree list of v2
            max_layer = min(len(lists_v1), len(lists_v2))
            dtw_dist[v1, v2] = {}
            for layer in range(0, max_layer):
                dist, path = fastdtw(lists_v1[layer], lists_v2[layer], radius=1, dist=dist_func)
                dtw_dict[v1, v2][layer] = dist
    return dtw_dict
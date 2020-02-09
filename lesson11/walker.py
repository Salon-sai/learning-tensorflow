import itertools
import math
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
# from tqdm import trange

from utils import *
from alias import *

class BiasedWalker:

    def __init__(self, idx2node, temp_path):
        self.idx2node = idx2node
        self.idx = list(range(len(self.idx2node)))
        self.temp_path = temp_path

    def simulate_walks(self, num_walks, walk_length, stay_prob=0.3, workers=1, verbose=0):
        layers_adj = pd.read_pickle(self.temp_path + "layers_adj.pkl")
        layers_alias = pd.read_pickle(self.temp_path + "layers_alias.pkl")
        layers_accept = pd.read_pickle(self.temp_path + "layers_accept.pkl")
        gamma = pd.read_pickle(self.temp_path + "gamma.pkl")
        walks = []
        initialLayer = 0

        nodes = self.idx
        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(self._simulate_walks)(nodes, num, walk_length, stay_prob, layers_adj, layers_accept, layers_alias, gamma) for num in partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))
        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, stay_prob, layers_adj, layers_accept, layers_alias, gamma):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                # 逐个节点进行随机游走
                walks.append(self._exec_random_walk(layers_adj, layers_accept, layers_alias, v, walk_length, gamma, stay_prob))
        return walks

    def _exec_random_walk(self, graphs, layers_accept, layers_alias, v, walk_length, gamma, stay_prob=0.3):
        
        initialLayer = 0
        layer = initialLayer

        path = []
        path.append(self.idx2node[v])
        while len(path) < walk_length:
            r = random.random()
            if (r < stay_prob): #　下一步在同一层当中游走
                v = chooseNeighbor(v, graphs, layers_alias, layers_accept, layer)
                path.append(self.idx2node[v])
            else: # 下一步前往上一层或者下一层
                r = random.random()
                try:
                    # x: 转移到H+1层的权值大小
                    x = math.log(gamma[layer][v] + math.e)
                    # p_moveup: 转移到H+1层的概率
                    p_moveup = (x / (x + 1))
                except:
                    print(layer, v)
                    raise ValueError()

                if (r > p_moveup and layer > initialLayer):
                    layer = layer - 1
                elif (r <= p_moveup and (layer + 1) in graphs and v in graphs[layer + 1]):
                    layer = layer + 1
        return path


def chooseNeighbor(v, graphs, layers_alias, layers_accept, layer):
    v_list = graphs[layer][v]
    # 使用alias　sample进行采样
    idx = alias_sample(layers_accept[layer][v], layers_alias[layer][v])
    v = v_list[idx]
    return v

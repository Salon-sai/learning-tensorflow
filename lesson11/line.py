
import random
import argparse
import networkx as nx
import numpy as np

import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Embedding, Input, Lambda
from tensorflow.python.keras.models import Model

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

def line_loss(y_true, y_pred):
    return - K.mean(K.log(K.sigmoid(y_true * y_pred)))

def create_model(numNodes, embedding_size, order='second'):
    v_i = Input(shape=(1,))
    v_j = Input(shape=(1,))

    first_emb = Embedding(numNodes, embedding_size, name='first_emb')
    second_emb = Embedding(numNodes, embedding_size, name='second_emb')
    context_emb = Embedding(numNodes, embedding_size, name='context_emb')

    v_i_emb = first_emb(v_i)
    v_j_emb = first_emb(v_j)

    v_i_emb_second = second_emb(v_i)
    v_j_context_emb = context_emb(v_j)

    first = Lambda(
        lambda x: tf.reduce_sum(x[0] * x[1], axis=-1, keep_dims=False), name='first_order')([v_i_emb, v_j_emb])
    second = Lambda(
        lambda x: tf.reduce_sum(x[0] * x[1], axis=-1, keep_dims=False), name='second_order')([v_i_emb_second, v_j_context_emb])

    if order == "first":
        output_list = [first]
    elif order == "second":
        output_list = [second]
    else:
        output_list = [first, second]

    model = Model(inputs=[v_i, v_j], outputs=output_list)

    return model, {'first': first_emb, 'second': second_emb}

def create_alias_table(area_ratio):
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])

        if  area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1

    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias

def alias_sample(accept, alias):
    N = len(accept)
    i = int(np.random.random() * N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]

class Line:

    def __init__(self, nx_G, batch_size, embedding_size, times=1, order='second', negative_ratio=5):
        self.G = nx_G

        self._embeddings = {}
        self.negative_ratio = negative_ratio
        self.order = order
        self.batch_size = batch_size
        self.embedding_size = embedding_size

        self.node_size = self.G.number_of_nodes()
        self.edge_size = self.G.number_of_edges()
        self.samples_per_epoch = self.edge_size * (1 + negative_ratio)
        self._gen_sampling_table()

        self.steps_per_epoch = ((self.samples_per_epoch - 1) // self.batch_size + 1) * times
        self.model, self.embedding_dict = create_model(self.node_size, self.embedding_size, self.order)
        self.model.compile("adam", line_loss)
        self.batch_it = self.batch_iter()

    def _gen_sampling_table(self):
        # create sampling table for vertex
        power = 0.75
        numNodes = self.node_size
        node_degree = np.zeros(numNodes)  # out degree

        for edge in self.G.edges():
            node_degree[edge[0]] += self.G[edge[0]][edge[1]].get('weight', 1.0)

        total_sum = sum([np.power(node_degree[i], power) for i in range(numNodes)])
        norm_prob = [float(np.power(node_degree[j], power)) / total_sum for j in range(numNodes)]

        self.node_accept, self.node_alias = create_alias_table(norm_prob)

        numEdges = self.G.number_of_edges()
        total_sum = sum([self.G[edge[0]][edge[1]].get('weight', 1.0) for edge in self.G.edges()])
        norm_prob = [self.G[edge[0]][edge[1]].get('weight', 1.0) * numEdges / total_sum for edge in self.G.edges()]

        self.edge_accept, self.edge_alias = create_alias_table(norm_prob)

    def batch_iter(self):
        edges = list(self.G.edges)
        data_size = self.G.number_of_edges()
        shuffle_indices = np.random.permutation(np.arange(data_size))

        # 判断是否需要shuffle
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        t = []
        sign = 0
        count = 0
        start_index = 0
        end_index = min(start_index + self.batch_size, data_size)

        while True:
            if mod == 0:
                h = []
                t = []
                for i in range(start_index, end_index):
                    if random.random() >= self.edge_accept[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
                sign = np.ones(len(h))
            else:
                sign = np.ones(len(h)) * -1
                t = []
                for i in range(len(h)):
                    t.append(alias_sample(self.node_accept, self.node_alias))

            if self.order == "all":
                yield ([np.array(h), np.array(t)], [sign, sign])
            else:
                yield ([np.array(h), np.array(t)], [sign])

            mod += 1
            mod %= mod_size

            if mod == 0:
                start_index = end_index
                end_index = min(start_index + self.batch_size, data_size)

            if start_index >= data_size:
                count += 1
                mod = 0
                h = []
                shuffle_indices = np.random.permutation(np.arange(data_size))
                start_index = 0
                end_index = min(start_index + self.batch_size, data_size)

    def get_embeddings(self):
        self._embeddings = {}
        if self.order == 'first':
            embeddings = self.embedding_dict['first'].get_weights()[0]
        elif self.order == 'second':
            embeddings = self.embedding_dict['second'].get_weights()[0]
        else:
            embeddings = np.hstack((self.embedding_dict['first'].get_weights()[0],
                                    self.embedding_dict['second'].get_weights()[0]))
        for i, embedding in enumerate(embeddings):
            self._embeddings[i] = embedding

        return self._embeddings

    def train(self, epoch=1, initial_epoch=0, verbose=1):
        hist = self.model.fit_generator(self.batch_it, epochs=epoch, initial_epoch=initial_epoch, verbose=verbose,
                                        steps_per_epoch=self.steps_per_epoch)
        return hist

def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")
    parser.add_argument('--input', nargs='?', default='data/Wiki_edgelist.txt', help='Input graph path')
    parser.add_argument('--output', nargs='?', default='emb/line_wiki.emb', help='Embeddings path')
    parser.add_argument('--label_file', nargs='?', default='data/wiki_labels.txt', help='Labels path')
    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')
    parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=20, help='Number of walks per source. Default is 10.')
    parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')
    parser.add_argument('--iter', default=2, type=int, help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')
    parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)
    parser.add_argument('--directed', dest='directed', action='store_true', help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)
    return parser.parse_args()

def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y

def plot_embeddings(embeddings, label_file):
    X, Y = read_node_label(label_file, skip_head=True)
    emb_list = []
    for k in X:
        emb_list.append(embeddings[int(k)])
    emb_list = np.array(emb_list)
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}

    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)
    plt.legend()
    plt.show()

def read_graph():
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float), ), create_using=nx.DiGraph)
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G

def main(args):
    nx_G = read_graph()
    line = Line(nx_G, batch_size=1024, embedding_size=128, order='second')
    line.train(epoch=2)
    _embeddings = line.get_embeddings()
    plot_embeddings(_embeddings, args.label_file)

if __name__ == "__main__":
    args = parse_args()
    main(args)
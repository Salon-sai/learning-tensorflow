

import numpy as np
import random

import argparse
import networkx as nx
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

class node2vec_walk():

    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            curr = walk[-1]
            cur_nbrs = sorted(G.neighbors(curr))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[curr][0], alias_nodes[curr][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, curr)][0], alias_edges[(prev, curr)][1])]
                    walk.append(next)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        walks = []
        nodes = list(G.nodes())

        print("Walk iteration...")

        for walk_iter in range(num_walks):
            print(f"{walk_iter + 1}/{num_walks}")
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length, node))
        return walks

    def get_alias_edge(self, src, dst):
        G = self.G
        p = self.p
        q = self.q
        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]["weight"])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / q)
        norm_cost = sum(unnormalized_probs)
        normalized_probs = [float(v) / norm_cost for v in unnormalized_probs]
        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        # 预处理转移概率
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]["weight"] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(v) / norm_const for v in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])


        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges



def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        # 记录小于均匀分布概率的Index
        if q[kk] > 1.0:
            larger.append(kk)
        else:
            smaller.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        # 记录index
        J[small] = large
        # 将small的补充满1后，算出剩余large的概率
        q[large] = q[small] + q[large] - 1
        # 若q[large]不等于1，则继续放入smaller和larger的数组中进行迭代
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    # 非均匀分布进行采样
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")
    parser.add_argument('--input', nargs='?', default='data/brazil-airports.edgelist', help='Input graph path')
    parser.add_argument('--output', nargs='?', default='emb/node2vec_wiki.emb', help='Embeddings path')
    parser.add_argument('--label_file', nargs='?', default='data/wiki_labels.txt', help='Labels path')
    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')
    parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=20, help='Number of walks per source. Default is 10.')
    parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')
    parser.add_argument('--iter', default=2, type=int, help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=1, help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')
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
        emb_list.append(embeddings[k])
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

def learning_walks(walks):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(args.output)
    return model

def main(args):
    nx_G = read_graph()
    G = node2vec_walk(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    model = learning_walks(walks)

    _embeddings = {}
    for word in nx_G.nodes():
        _embeddings[str(word)] = model.wv[str(word)]

    plot_embeddings(_embeddings, args.label_file)

if __name__ == "__main__":
    args = parse_args()
    main(args)




import numpy as np
import random

import argparse
import networkx as nx
from gensim.models import Word2Vec

from utils import read_node_label, plot_embeddings

class deepwalk():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def random_walk(self, walk_length, start_node):
        G = self.G
        walk = [start_node]
        for _ in range(walk_length):
            curr = walk[-1]
            nbr_nodes = sorted(G.neighbors(curr))
            walk.append(random.choice(nbr_nodes))
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
                walks.append(self.random_walk(walk_length, node))
        return walks

def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")
    parser.add_argument('--input', nargs='?', default='./data/Wiki_edgelist.txt', help='Input graph path')
    parser.add_argument('--output', nargs='?', default='emb/deepwalk_wiki.emb', help='Embeddings path')
    parser.add_argument('--label_file', nargs='?', default='./data/wiki_labels.txt', help='Labels path')
    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')
    parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=20, help='Number of walks per source. Default is 10.')
    parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')
    parser.add_argument('--iter', default=2, type=int, help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=1., help="P of Param")
    parser.add_argument('--q', type=float, default=1., help="Q of Param")
    parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)
    parser.add_argument('--directed', dest='directed', action='store_true', help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)
    return parser.parse_args()

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
    G = deepwalk(nx_G, args.directed, args.p, args.q)
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    model = learning_walks(walks)

    _embeddings = {}
    for word in nx_G.nodes():
        _embeddings[str(word)] = model.wv[str(word)]

    plot_embeddings(_embeddings, args.label_file)

if __name__ == "__main__":
    args = parse_args()
    main(args)


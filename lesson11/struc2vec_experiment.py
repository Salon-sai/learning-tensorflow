
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

from struc2vec import Struc2Vec
from utils import evaluate_embeddings, plot_embeddings

if __name__ == "__main__":
    G = nx.read_edgelist('./data/Wiki_edgelist.txt', create_using=nx.DiGraph(), nodetype=None,
                         data=[('weight', int)])
    model = Struc2Vec(G, 20, 80, workers=4, verbose=40, opt3_num_layers=5)
    model.train(embed_size=256)
    embeddings = model.get_embeddings()    

    evaluate_embeddings(embeddings, "./data/wiki_labels.txt")
    plot_embeddings(embeddings, "./data/wiki_labels.txt")

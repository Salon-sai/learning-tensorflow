# -*- coding:utf-8 -*-

import time
import argparse

import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy import sparse

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l1_l2

from utils import read_node_label, plot_embeddings, evaluate_embeddings

class SDNE(object):

    def __init__(self, graph, hidden_size=[32, 16], alpha=1e-6, beta=5., nu1=1e-5, nu2=1e-4):
        self.graph = graph
        self.node_size = self.graph.number_of_nodes()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2

        # 获取邻接矩阵，拉普拉斯矩阵
        self.A, self.L = self._sparse_create_A_L(self.graph)
        self.inputs = [self.A, self.L]
        self._embeddings = {}
        self.model, self.emb_model = self.model_graph(hidden_size=hidden_size)
        self.model.compile("adam", [self.loss_2(self.beta), self.loss_1(self.alpha)])

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1):
        if batch_size >= self.node_size:
            if batch_size > self.node_size:
                print('batch_size({0}) > node_size({1}),set batch_size = {1}'.format(
                    batch_size, self.node_size))
                batch_size = self.node_size
            return self.model.fit([self.A, self.L], [self.A, self.L], batch_size=batch_size, epochs=epochs, initial_epoch=initial_epoch, verbose=verbose, shuffle=False)
        else:
            steps_per_epoch = (self.node_size - 1) // batch_size + 1
            hist = History()
            hist.on_train_begin()
            logs = {}
            for epoch in range(initial_epoch, epochs):
                start_time = time.time()
                losses = np.zeros(3)
                for i in range(steps_per_epoch):
                    index = np.arange(i * batch_size, min((i + 1) * batch_size, self.node_size))

                    train_A = self.A[index, :]  # batch_size * node_size
                    if sparse.issparse(train_A):
                        train_A = train_A.toarray()
                    train_L = self.L[index][:, index] # L矩阵是一个对角矩阵
                    if sparse.issparse(train_L):
                        train_L = train_L.toarray()

                    inp = [train_A, train_L]
                    batch_losses = self.model.train_on_batch(inp, inp)
                    losses += batch_losses
                losses = losses / steps_per_epoch
                logs['loss'] = losses[0]
                logs['2nd_loss'] = losses[1]
                logs['1st_loss'] = losses[2]
                epoch_time = int(time.time() - start_time)
                hist.on_epoch_end(epoch, logs)
                if verbose > 0:
                    print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                    print('{0}s - loss: {1: .4f} - 2nd_loss: {2: .4f} - 1st_loss: {3: .4f}'.format(
                        epoch_time, losses[0], losses[1], losses[2]))
            return hist

    def loss_1(self, alpha):
        def loss_1st(y_true, y_pred):
            L = y_true
            Y = y_pred
            batch_size = tf.to_float(K.shape(L)[0])
            return alpha * 2 * tf.linalg.trace(tf.matmul(tf.matmul(Y, L, transpose_a=True), Y)) / batch_size
        return loss_1st

    def loss_2(self, beta):
        def loss_2nd(y_true, y_pred):
            b_ = np.ones_like(y_true)
            b_[y_true != 0] = beta
            x = K.square((y_true - y_pred) * b_)
            t = K.sum(x, axis=-1, )
            return K.mean(t)
        return loss_2nd

    def define_graph(self, l1=1e-5, l2=1e-4, hidden_size=[256, 128]):
        a = tf.placeholder(dtype=tf.float32, shape=[None, self.node_size], name="node_vec")
        l = tf.placeholder(dtype=tf.float32, shape=[None, None], name="l")

        x = a
        for unit in hidden_size:
            x = tf.layers.dense(x, unit, activation="relu")
        
        y = tf.identity(x, name="y")

        for i in reversed(range(len(hidden_size) - 1)):
            x = tf.layers.dense(x, hidden_size[i], activation="relu")

        x_mat = tf.identity(x, name="x_mat")

    def model_graph(self, l1=1e-5, l2=1e-4, hidden_size=[256, 128]):
        a = Input(shape=(self.node_size,))
        l = Input(shape=(None,))
        output = a
        for i in range(len(hidden_size)):
            if i == len(hidden_size) - 1:
                output = Dense(hidden_size[i], activation="relu", kernel_regularizer=l1_l2(l1, l2), name="1st")(output)
            else:
                output = Dense(hidden_size[i], activation="relu", kernel_regularizer=l1_l2(l1, l2))(output)
        # 记录一阶向量
        y = output
        for i in reversed(range(len(hidden_size) - 1)):
            output = Dense(hidden_size[i], activation="relu", kernel_regularizer=l1_l2(l1, l2))(output)

        a_ = Dense(self.node_size, activation="relu", name="2nd")(output)
        model = Model(inputs=[a, l], outputs=[a_, y])
        emb = Model(inputs=a, outputs=y)
        return model, emb

    def get_embeddings(self):
        self._embeddings = {}
        embeddings = self.emb_model.predict(self.A, batch_size=self.node_size)
        for i, embedding in enumerate(embeddings):
            self._embeddings[i] = embedding
        return self._embeddings

    def _create_A_L(self, graph):
        A = np.zeros((self.node_size, self.node_size))
        A_ = np.zeros((self.node_size, self.node_size))

        for edge in graph.edges():
            v1, v2 = edge
            weight = graph[v1][v2].get("weight", 1)
            A[int(v1)][int(v2)] = weight
            A_[int(v1)][int(v2)] = weight
            A_[int(v2)][int(v1)] = weight
            
        D = np.zeros_like(A)
        for i in range(self.node_size):
            D[i][i] = np.sum(A_[i])
        L = D - A_
        return A, L

    def _sparse_create_A_L(self, graph):
        # A = np.zeros((self.node_size, self.node_size))
        # TODO: 只针对权值为1的邻接图
        A_ = nx.adjacency_matrix(graph, sorted(graph.nodes(), key=lambda x: int(x))).astype(np.int16)
        A = ((A_ + A_.T) != 0).astype(np.int16)
        D = sparse.diags(np.array(A.sum(0)).flatten()).astype(np.int16)
        L = (D - A).astype(np.int16)
        return A_, L

def parse_args():
    parser = argparse.ArgumentParser(description="SDNE parameter")
    parser.add_argument('--input', nargs='?', default='data/Wiki_edgelist.txt', help='Input graph path')
    parser.add_argument('--output', nargs='?', default='emb/sdne_wiki.emb', help='Embeddings path')
    parser.add_argument('--label_file', nargs='?', default='data/wiki_labels.txt', help='Labels path')
    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')
    return parser.parse_args()

def main(args):
    nx_G = nx.read_edgelist(args.input, create_using=nx.DiGraph(), nodetype=None, data=[("weight", int)])
    model = SDNE(nx_G, hidden_size=[512, 256, 256])
    model.train(batch_size=2000, epochs=50, verbose=2)

    embeddings = model.get_embeddings()
    embeddings = {str(k): embeddings[k] for k in embeddings.keys()}
    evaluate_embeddings(embeddings, args.label_file)
    plot_embeddings(embeddings, args.label_file)

if __name__ == "__main__":
    args = parse_args()
    main(args)


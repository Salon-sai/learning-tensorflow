# -*- coding: utf-8 -*-
import random

from lesson9.neuro_network import random_clamped

class Genome(object):
    def __init__(self, score, network_weights):
        self.score = score
        self.network_weights = network_weights

class Generation(object):
    def __init__(self, score_sort=-1, mutation_rate=0.05, mutation_range=2, elitism=0.2, population=50,
                 random_behaviour=0.1, n_child=1):
        self.genomes = []
        self._score_sort = score_sort
        self._mutation_rate = mutation_rate
        self._mutation_range = mutation_range
        self._elitism = elitism
        self._population = population
        self._random_behaviour = random_behaviour
        self._n_child = n_child

    def add_genome(self, genome):
        i = 0
        for i in range(len(self.genomes)):
            if self._score_sort < 0:
                if genome.score > self.genomes[i].score:
                    break
            else:
                if genome.score < self.genomes[i].score:
                    break
        self.genomes.insert(i, genome)

    def breed(self, genome1, genome2, n_child):
        """
        breed children between genome1 and genome2
        :param genome1:
        :param genome2:
        :param n_child: generate the number of children
        :return:
        """
        datas = []
        for n in range(n_child):
            # data = genome1
            data = Genome(0, {'network': genome1.network_weights['network'][:],
                              'weights': genome1.network_weights['weights'][:]})
            for i in range(len(genome2.network_weights['weights'])):
                # crossover values
                if random.random() <= 0.7:
                    data.network_weights['weights'][i] = genome2.network_weights['weights'][i]
            for i in range(len(data.network_weights['weights'])):
                # mutate values
                if random.random() <= self._mutation_rate:
                    data.network_weights['weights'][i] += random.random() * self._mutation_rate * 2 - self._mutation_range
                datas.append(data)
        return datas

    def generate_next_generation(self):
        nexts = []  # the weights of genes
        for i in range(round(self._elitism * self._population)):
            if len(nexts) < self._population:
                nexts.append(self.genomes[i].network_weights)

        for i in range(round(self._random_behaviour * self._population)):
            n = self.genomes[0].network_weights
            for k in range(len(n['weights'])):
                # generate all values of weights
                n['weights'][k] = random_clamped()
            if len(nexts) < self._population:
                nexts.append(n)
        max_n = 0
        while True:
            for i in range(max_n):
                childs = self.breed(self.genomes[i], self.genomes[max_n], self._n_child if self._n_child > 0 else 1)
                for c in range(len(childs)):
                    nexts.append(childs[c].network_weights)
                    if len(nexts) >= self._population:
                        return nexts
            max_n += 1
            if max_n >= len(self.genomes) - 1:
                max_n = 0

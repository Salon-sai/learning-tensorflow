# -*- coding: utf-8 -*-

from lesson9.neuro_network import NeuroNetwork
from lesson9.GA import Generation, Genome

class Generations(object):
    def __init__(self, population=50, network_size=None):
        self.generations = []
        self._population = population
        if network_size is None:
            self._network_size = [4, [16], 1]
        else:
            self._network_size = network_size

    def first_generation(self):
        out = []
        # init the population with neuro-network
        for i in range(self._population):
            nn = NeuroNetwork()
            nn.init_neuro_network(self._network_size[0], self._network_size[1], self._network_size[2])
            out.append(nn.get_weight())
        self.generations.append(Generation())
        return out

    def next_generation(self):
        if len(self.generations) == 0:
            return False
        gen = self.generations[-1].generate_next_generation()
        self.generations.append(Generation())
        return gen

    def add_genome(self, genome):
        if len(self.generations) == 0:
            return False
        return self.generations[-1].add_genome(genome)


class NeuroEvolution(object):
    def __init__(self):
        self.generations = Generations()

    def restart(self):
        self.generations = Generations()

    def next_generation(self, low_historic=False, historic=0):
        networks = []
        # get the weights of networks
        if len(self.generations.generations) == 0:
            networks = self.generations.first_generation()
        else:
            networks = self.generations.next_generation()

        nn = []
        for i in range(len(networks)):
            n = NeuroNetwork()
            n.set_weight(networks[i])
            nn.append(n)

        if low_historic:
            if len(self.generations.generations) >= 2:
                genomes = self.generations.generations[len(self.generations.generations) - 2].genomes
                for i in range(genomes):
                    genomes[i].network = None

        if historic != -1:
            if len(self.generations.generations) > historic + 1:
                del self.generations.generations[0:len(self.generations.generations) - (historic + 1)]
        return nn

    def network_score(self, score, network):
        self.generations.add_genome(Genome(score, network.get_weight()))



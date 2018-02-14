# -*- coding: utf-8 -*-

import random
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def random_clamped():
    return random.random() * 2 - 1

class Neuron(object):
    def __init__(self):
        self.bias = 0
        self.weights = []

    def init_weight(self, n):
        self.weights = []
        for i in range(n):
            self.weights.append(random_clamped())

    # def __repr__(self):
    #     return 'Neuron weight size: {}, biase value: {}'.format(len(self.weights), self.bias)

class Layer(object):

    def __init__(self, index):
        self.index = index
        self.neurons = []

    def init_neurons(self, n_output, n_input):
        self.neurons = []
        for i in range(n_output):
            neuron = Neuron()
            neuron.init_weight(n_input)
            self.neurons.append(neuron)

    # def __repr__(self):
    #     return 'Layer ID: {}, Layer neuron size: {}'.format(self.index, len(self.neurons))

class NeuroNetwork(object):

    def __init__(self):
        self._layers = []

    def init_neuro_network(self, input, hiddens, output):
        index = 0
        previous_neurons = 0
        layer = Layer(index)
        layer.init_neurons(input, previous_neurons)
        previous_neurons = input
        self._layers.append(layer)
        index += 1
        for i in range(len(hiddens)):
            layer = Layer(index)
            layer.init_neurons(hiddens[i], previous_neurons)
            previous_neurons = hiddens[i]
            self._layers.append(layer)
            index += 1
        layer = Layer(index)
        layer.init_neurons(output, previous_neurons)
        self._layers.append(layer)

    def get_weight(self):
        data = {'network': [], 'weights': []}
        for layer in self._layers:
            data['network'].append(len(layer.neurons))
            for neuron in layer.neurons:
                for weight in neuron.weights:
                    data['weights'].append(weight)
        return data

    def set_weight(self, data):
        previous_neurous = 0
        index = 0
        index_weights = 0

        self._layers = []
        for num_output in data['network']:
            layer = Layer(index)
            layer.init_neurons(num_output, previous_neurous)
            for j in range(num_output):
                for k in range(len(layer.neurons[j].weights)):
                    layer.neurons[j].weights[k] = data['weights'][index_weights]
                    index_weights += 1
            previous_neurous = num_output
            index += 1
            self._layers.append(layer)

    def feed_forward(self, inputs):
        """
        input the status
        :param inputs:
        :return:
        """
        # input the status for input neurons
        for i in range(len(inputs)):
            self._layers[0].neurons[i].biase = inputs[i]

        prev_layer = self._layers[0]
        for i in range(len(self._layers)):
            if i == 0:
                continue
            for j in range(len(self._layers[i].neurons)):
                # this loop get each neuron of current layer
                sum = 0
                for k in range(len(prev_layer.neurons)):
                    # loop previous of output to get calculate the linear result in j-th neuron
                    # calculate the product between weights and previous output
                    sum += prev_layer.neurons[k].biase * self._layers[i].neurons[j].weights[k]
                # calculate sigmoid with linear result
                self._layers[i].neurons[j].biase = sigmoid(sum)
            prev_layer = self._layers[i]
        out = []
        last_layer = self._layers[-1]
        for i in range(len(last_layer.neurons)):
            out.append(last_layer.neurons[i].biase)
        return out

    def print_info(self):
        for layer in self._layers:
            print(layer)


import numpy as np
import math

### HYPERPARAMETERS ###

# Network Params #

NODES = [4, 4]

# PSO Params #


class Network:

    def __init__(self, nodes):

        self.nodes = nodes

        self.inputWeights = np.random.randn(1, self.nodes[0])

        self.w = {}

        for x in range(0, len(self.nodes) - 1):

            arr = np.random.randn(nodes[x], nodes[x+1])
            self.w["weights{0}".format(x+1)] = arr

        self.outputWeights = np.random.randn(nodes[len(nodes) - 1], 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, input):

        self.layer = self.sigmoid(np.dot(input, self.inputWeights))

        for x in range(0, len(self.nodes) - 1):
            if x == 0:
                self.layer = self.sigmoid(np.dot(self.layer, self.w["weights{0}".format(x+1)]))

            else:
                self.layer = self.sigmoid(np.dot(self.layer, self.w["weights{0}".format(x+1)]))

        self.output = self.sigmoid(np.dot(self.layer, self.outputWeights))

        return self.output


net = Network(NODES)
net.feedforward(1)

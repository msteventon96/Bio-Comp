import numpy as np
import math

### HYPERPARAMETERS ###

FILE = "1in_cubic.txt"

# Network Params #

NODES = [4, 4]

# PSO Params #

SWARMSIZE = 50
VELOCITY = 0.5
PERSONAL_BEST = 1
INFORM_BEST = 1.5
GLOBAL_BEST = 1.5
STEPSIZE = 1
ITERATIONS = 1000

class Network:

    def __init__(self, nodes):

        self.nodes = nodes
        self.velocity = np.random.randint(5)
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

    def mse(self, pred, target):

        return (np.square(target - pred)).mean()


def train():

    particles = []

    for x in range(0, SWARMSIZE - 1):

        net = Network(NODES)
        particles.append(net)

    globalBest = None
    i = 0

    while i < ITERATIONS:

        for x in range(0, len(particles)):

            if globalBest is None or Fitness(x) > Fitness(globalBest):

                globalBest = particles[x]

inputs = []
outputs = []

file = open(FILE)

for line in file:

    split = line.split(" ")

    inputs.append(split[0])

    for x in range(1, len(split)):
        if split[x] != "":
            outputs.append(split[x].rstrip())


print(inputs)
print(outputs)

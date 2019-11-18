import numpy as np
import random
import math

### HYPERPARAMETERS ###

FILE = "1in_linear.txt"

# Network Params #

INPUT = 1
NODES = [3]

# PSO Params #

SWARMSIZE = 75
VELOCITY = 1
PERSONAL_BEST = 1.2
INFORM_BEST = 1.4
GLOBAL_BEST = 1.4
STEPSIZE = 1
ITERATIONS = 100

class Network:

    def __init__(self, input, nodes):

        self.input = input
        self.nodes = nodes

        self.informant = []

        self.velocity = np.random.randint(5)

        self.inputWeights = np.random.rand(self.nodes[0], input)
        self.weights = self.inputWeights
        self.w = {}

        for x in range(0, len(self.nodes) - 1):

            for y in range(0, nodes[x+1]):

                self.weights = np.append(self.weights, np.random.rand(nodes[x], 1), axis=0)

        self.outputWeights = np.random.rand(nodes[len(nodes) - 1], 1)
        self.weights = np.append(self.weights, self.outputWeights, axis=0)

        self.fittestWeights = self.weights
        self.fittestError = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tan(self, x):
        return np.tanh(x)

    def cos(self, x):
        return np.cos(x)

    def gaussian(self, x):
        return np.exp(- np.divide(np.square(x), 2))

    def feedforward(self, input):

        numinputs = INPUT*NODES[0]
        numoutputs = NODES[len(NODES) - 1]

        temp = np.random.randn(numinputs, 1)

        for x in range(0, numinputs):
            temp[x] = self.weights[x]

        temp = np.squeeze(np.asarray(temp))

        layer = []

        for x in range(numinputs):
            layer.append(input)

        self.layer = self.tan(np.dot(input, temp))

        if len(NODES) > 1:
            temp = np.random.randn(len(self.weights) - (numinputs + numoutputs), 1)

            for x in range(numinputs, len(self.weights) - (numinputs + numoutputs)):

                temp[x] = self.weights[x]

            temp = np.squeeze(np.asarray(temp))

            layer = []

            temp2 = []
            for y in range(NODES[1]):
                for x in range(NODES[1]):
                    temp2.append(temp[x])
                    if len(temp2) == NODES[1]:
                        layer.append(np.dot(self.layer, temp2))
                        temp2 = []

        temp = np.random.randn(numoutputs, 1)

        for x in range(0, numoutputs):

            temp[x] = self.weights[x]

        temp = np.squeeze(np.asarray(temp))

        self.output = np.dot(layer, temp)

        return self.output

    def mse(self, pred, target):

        error = (np.square(np.subtract(target, pred))).mean()

        if self.fittestError is None or error < self.fittestError:
            self.fittestError = error
            self.fittestWeights = self.weights

        return error


def train(inputs, outputs):

    particles = []

    for x in range(0, SWARMSIZE - 1):

        net = Network(INPUT, NODES)
        particles.append(net)

    for x in range(0, len(particles)):

        for y in range(5):

            int = np.random.randint(len(particles))
            particles[x].informant.append(particles[int])

    globalBest = None


    for i in range(0, ITERATIONS):

        for x in range(0, len(particles)):

            predicted = []
            for y in range(len(inputs)):
                predicted.append(particles[x].feedforward(inputs[y]))

            loss = particles[x].mse(predicted, outputs)

            if globalBest is None:

                globalBest = particles[x]

            elif loss < globalBest.fittestError:

                globalBest = particles[x]

        if globalBest is not None and i % 10 == 0:
            print("Epoch: {}, Best Error: ".format(i), globalBest.fittestError)

        for x in range(0, len(particles)):

            nnBestWeights = particles[x].fittestWeights

            bestinformantweights = None
            besterror = None
            for y in range(len(particles[x].informant)):

                if bestinformantweights is None or particles[x].informant[y].fittestError < besterror:

                    bestinformantweights = particles[x].informant[y].fittestWeights
                    besterror = particles[x].informant[y].fittestError

            bestglobalweights = globalBest.fittestWeights

            if particles[x] is not globalBest:
                for z in range(len(particles[x].weights)):

                    b = random.uniform(0.0, PERSONAL_BEST)
                    c = random.uniform(0.0, INFORM_BEST)
                    d = random.uniform(0.0, GLOBAL_BEST)

                    particles[x].weights[z] = (VELOCITY*particles[x].weights[z]) + \
                                              b*(nnBestWeights[z] - particles[x].weights[z]) + \
                                              c*(bestinformantweights[z] - particles[x].weights[z]) + \
                                              d*(bestglobalweights[z] - particles[x].weights[z])

        # for x in range(len(particles)):
        #
        #     particles[x].weights = particles[x].weights + STEPSIZE*particles[x].weights

    return globalBest

inputs = []
outputs = []

file = open(FILE)

if FILE == "1in_linear.txt":

    for line in file:

        split = line.split('\t')

        inputs.append(split[0])

        for x in range(1, len(split)):
            if split[x] != "":
                outputs.append(split[x].rstrip())

elif FILE.__contains__("1in"):

    for line in file:

        split = line.split(" ")

        inputs.append(split[0])

        for x in range(1, len(split)):
            if split[x] != "":
                outputs.append(split[x].rstrip())

else:
    print("2 inputs not implemented yet, please select 1 input")

inputs = list(map(float, inputs))
outputs = list(map(float, outputs))


best = train(inputs, outputs)

for x in range(len(inputs)):
    print(inputs[x])
    print(best.feedforward(inputs[x]))


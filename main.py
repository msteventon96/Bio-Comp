import numpy as np
import random

### HYPERPARAMETERS ###

FILE = "1in_linear.txt"     # File to be used

# Network Params #

INPUT = 1                   # Number of inputs
NODES = [3]                 # Layers and nodes per layer defined in an array e.g. [3,3] for 2 layers, 3 nodes each

# PSO Params #

SWARMSIZE = 75              # Amount of particles within the algorithm
VELOCITY = 1                # Inertia coefficient of each particle
PERSONAL_BEST = 1.2         # Coefficient for each particles fittest weights
INFORM_BEST = 1.4           # Coefficient for each particles informants fittest weights
GLOBAL_BEST = 1.4           # Coefficient for the global best particles fittest weights
STEPSIZE = 1                # Unused in algorithm
ITERATIONS = 100            # Total number of iterations before training completion


class Network:

    def __init__(self, input, nodes):
        """
        :param input: Number of inputs
        :param nodes: Number of layers and nodes per layer

        Initialises all the weights within the ANN and stores them in self.weights as a list that the program can
        select from later
        """

        self.input = input
        self.nodes = nodes

        self.informant = []         # Placeholder for adding informants for each ANN created

        self.inputWeights = np.random.rand(self.nodes[0], input)   # List of random weights between 0-1

        self.weights = self.inputWeights

        for x in range(0, len(self.nodes) - 1):

            for y in range(0, nodes[x+1]):
                # Append more random weights for each layer the user wants

                self.weights = np.append(self.weights, np.random.rand(nodes[x], 1), axis=0)

        self.outputWeights = np.random.rand(nodes[len(nodes) - 1], 1)
        self.weights = np.append(self.weights, self.outputWeights, axis=0)  # Add the output weights

        self.fittestWeights = self.weights   # Placeholder for the ANNs best weights during training
        self.fittestError = None             # Placeholder for the ANNs lowest error during training

    """
    Definitions of all the activation functions tested
    """
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tan(self, x):
        return np.tanh(x)

    def cos(self, x):
        return np.cos(x)

    def gaussian(self, x):
        return np.exp(- np.divide(np.square(x), 2))

    def feedforward(self, input):
        """
        :param input: An input taken from the file e.g. -1.0
        :return: The predicted output from the ANN

        Takes an input, creates the dot product for every neuron in the network, feeding the input through until
        it creates 1 output.
        """

        numinputs = INPUT*NODES[0]
        numoutputs = NODES[len(NODES) - 1]

        temp = np.random.randn(numinputs, 1)

        for x in range(0, numinputs):
            temp[x] = self.weights[x]

        temp = np.squeeze(np.asarray(temp))

        layer = []

        for x in range(numinputs):
            layer.append(input)

        self.layer = self.tan(np.dot(input, temp))   # Dot product of the first layer, usually a list

        if len(NODES) > 1:   # If hidden layers > 1

            """ Has to apply the previous dot products individually to each weight in the next layer"""

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

        self.output = np.dot(layer, temp)   # Takes the previous weights and creates a single output

        return self.output

    def mse(self, pred, target):
        """
        :param pred: Predicted outputs of the function
        :param target: Actual outputs of the function
        :return: The mean square loss of the ANN
        """

        error = (np.square(np.subtract(target, pred))).mean()

        if self.fittestError is None or error < self.fittestError:
            self.fittestError = error
            self.fittestWeights = self.weights

        return error


def train(inputs, outputs):
    """
    :param inputs: List of inputs taken from selected file
    :param outputs: List of outputs taken from selected file
    :return: The best trained ANN which is the ANN that got its error closest to 0

    This is the PSO algorithm.
    """

    particles = []          # Initialize the particle set i.e. The list of ANNs

    for x in range(0, SWARMSIZE - 1):

        net = Network(INPUT, NODES)
        particles.append(net)           # Append each initiated ANN each with random weights

    for x in range(0, len(particles)):

        for y in range(5):

            # For every particle, it selects 5 random other particles to be its informant

            int = np.random.randint(len(particles))
            particles[x].informant.append(particles[int])

    globalBest = None

    for i in range(0, ITERATIONS):

        """ Updating the Global Best """

        for x in range(0, len(particles)):

            predicted = []
            for y in range(len(inputs)):
                predicted.append(particles[x].feedforward(inputs[y]))

            loss = particles[x].mse(predicted, outputs)         # Current loss of selected particle

            if globalBest is None:

                globalBest = particles[x]

            elif loss < globalBest.fittestError:

                globalBest = particles[x]

        if globalBest is not None and i % 10 == 0:
            print("Epoch: {}, Best Error: ".format(i), globalBest.fittestError)

        for x in range(0, len(particles)):

            nnBestWeights = particles[x].fittestWeights         # Selects the best weights of the selected particle

            bestinformantweights = None         # Selects the best weights of the selected particles informants
            besterror = None
            for y in range(len(particles[x].informant)):

                if bestinformantweights is None or particles[x].informant[y].fittestError < besterror:

                    bestinformantweights = particles[x].informant[y].fittestWeights
                    besterror = particles[x].informant[y].fittestError

            bestglobalweights = globalBest.fittestWeights           # Selects the best weights of the global best

            if particles[x] is not globalBest:
                for z in range(len(particles[x].weights)):

                    b = random.uniform(0.0, PERSONAL_BEST)          # Random number for personal coefficient
                    c = random.uniform(0.0, INFORM_BEST)            # Random number for informant coefficient
                    d = random.uniform(0.0, GLOBAL_BEST)            # Random number for global coefficient

                    """ Update weights """

                    particles[x].weights[z] = (VELOCITY*particles[x].weights[z]) + \
                                              b*(nnBestWeights[z] - particles[x].weights[z]) + \
                                              c*(bestinformantweights[z] - particles[x].weights[z]) + \
                                              d*(bestglobalweights[z] - particles[x].weights[z])

        # for x in range(len(particles)):
        #
        #     particles[x].weights = particles[x].weights + STEPSIZE*particles[x].weights

    return globalBest


""" Takes inputs and outputs from files and seperates them into different lists """

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

""" Trains a set of ANNs and get the best """
best = train(inputs, outputs)

""" Test the ANN with inputs """
for x in range(len(inputs)):
    print("Input: ", inputs[x], " Predicted: ", best.feedforward(inputs[x]))


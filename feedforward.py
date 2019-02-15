import numpy as np
import math

class feedforward:
    def __init__(self, dim):
        self.dim = dim
        self.softmax = []
        self.loss = 0
        self.activations = []
        self.outputs = []
        weights = []
        bias = []
        for index, i in enumerate(dim[:-1]):
            bias.append(np.random.rand(dim[index + 1]))
            weights.append(np.random.rand(dim[index + 1], i))
        self.bias = bias
        self.weights = weights

    # x is np array
    def train(self, x):
        cur = x
        for index, layer in enumerate(self.weights[:-1]):
            cur = np.matmul(layer, cur) + self.bias[index]
            self.outputs.append(cur)
            cur = self.leakyRelu(cur)
            self.activations.append(cur)
        cur = np.matmul(self.weights[-1], cur)
        res = self.softMax(cur)
        return res

    # v is a np vector
    def leakyRelu(self, v):
        l_relu = lambda x: max([.001 * x, x])
        v_relu = np.vectorize(l_relu)
        return v_relu(v)

    def softMax(self, v):
        vals = [math.exp(i) for i in v]
        s = np.sum(vals)
        self.softmax = [math.exp(i) / s for i in v]

    def logLoss(self, gt, predict):
        self.loss = -1 * np.log(np.dot(gt, predict))

    def bProp(self, less, gt, predict):
        updates = []
        updates.append(self.gradDescSoftLog())
        for index, weight in enumerate(reversed(self.weights[:-1]])):
            updates.append(gradDesc(weight, updates[-1], self.activations[-(index + 1)]))


    def gradDesc(self, weight, prevGrads, activations):
        gradients = np.empty(shape=weight.shape)
       #sum down the columns 
        for i in gradients:



    def gradDescSoftLog(self):
        for i in self.weights[-1]:
            for j in i:
                gradients[i][j] = (self.softmax[i] - gt[i]) * predict[j]
        return gradients



    def printShape(self):
        for weight in self.weights:
            print(weight.shape)
        for i in self.bias:
            print(i.shape)

shape = [6, 4, 2]
print('shape ' + str(shape))
net = feedforward(shape)
net.printShape()
x = np.random.rand(6)
y = np.array([1,0])
res = net.train(x)
print(x)
print(y)
print(res)
l = net.logLoss(y, res)
print(l)


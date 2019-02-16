import numpy as np
import math

class feedforward:
    def __init__(self, dim):
        self.dim = dim
        self.softmax = []
        self.loss = 0
        self.activations = []
        self.learningRate = .01
        self.outputs = []
        self.leak = -1 * 0.001
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
        self.activations.append(cur)
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
        l_relu = lambda x: max([self.leak * x, x])
        v_relu = np.vectorize(l_relu)
        return v_relu(v)

    def softMax(self, v):
        vals = [math.exp(i) for i in v]
        s = np.sum(vals)
        self.softmax = [math.exp(i) / s for i in v]
        return self.softmax

    def logLoss(self, gt, predict):
        self.loss = -1 * np.log(np.dot(gt, predict))
        return self.loss

    def bProp(self, loss, gt, predict):
        updates = []
        updates.append(self.gradDescSoftLog(self.weights[-1], gt, self.activations[-1]))
        for index, weight in enumerate(reversed(self.weights[:-1])):
            updates.append(self.gradDesc(weight, updates[-1], self.activations[-(index + 1)], self.outputs[-(index + 1)]))
        for i, update in enumerate(reversed(updates)):
            self.updateWeights(self.weights[i], update)


    def gradDesc(self, weight, prevGrads, activations, outputs):
        w_gradients = np.empty(shape=weight.shape)
        b_gradients = np.empty(shape=weight.shape)
        m_sum = np.sum(prevGrads, axis=0)
        for i_index, i in enumerate(w_gradients):
            for j_index, j in enumerate(i):
                d_activation = activations[i_index] if activations[i_index] > 0 else activations[i_index] * self.leak
                j = d_activation * m_sum[i_index] * outputs[i_index]
        return w_gradients

    def updateWeights(self, weights, updates):
        for i_index, i in enumerate(updates):
            for j_index, j in enumerate(i):
                weights[i_index][j_index] += self.learningRate * j


    def gradDescSoftLog(self, weight, gt, activations):
        w_gradients = np.empty(shape=weight.shape)
        for i_index, i in enumerate(w_gradients):
            for j_index, j in enumerate(i):
                j = (self.softmax[i_index] - gt[i_index]) * activations[j_index]
        return w_gradients

    def printShape(self):
        for weight in self.weights:
            print(weight.shape)
        for i in self.bias:
            print(i.shape)

shape = [6, 4, 2]
net = feedforward(shape)
print('prior weights', net.weights)
x = np.random.rand(6)
y = np.array([1,0])
print('train data', x)
print('gt', y)
res = net.train(x)
loss = net.logLoss(y, res)
net.bProp(loss, y, res)
print('post weights', net.weights)

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
            weights.append(np.random.rand(dim[index + 1], i))
        for i in dim[1:-1]:
            bias.append(np.random.rand(i))
        self.bias = bias
        self.weights = weights

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
        print('activations', self.activations)
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
        w_updates = []
        b_updates = []
        softLogUpdates = self.gradDescSoftLog(self.weights[-1], self.bias[-1], gt, self.activations[-1])

        w_updates.append(softLogUpdates[0])
        b_updates.append(softLogUpdates[1])
        for index, weight in enumerate(reversed(self.weights[:-1])):
            updates = self.gradWDesc(weight, w_updates[-1], self.activations[-(index + 2)], self.outputs[-(index + 1)])
            w_updates.append(updates)

        for index, bias in enumerate(reversed(self.bias[:-1])):
            updates = self.gradBDesc(bias, w_updates[-1], self.activations[-(index + 2)])
            b_updates.append(updates)

        for i, update in enumerate(reversed(w_updates)):
            self.updateWeights(self.weights[i], update)

        for i, update in enumerate(reversed(b_updates)):
            self.updateBias(self.bias[i], update)


    def gradWDesc(self, weight, prevGrads, activations, outputs):
        w_gradients = np.empty(shape=weight.shape)
        w_sum = np.sum(prevGrads, axis=0)
        for i_index, i in enumerate(w_gradients):
            for j_index, j in enumerate(i):
                d_activation = activations[j_index] if activations[j_index] > 0 else activations[j_index] * self.leak
                j = d_activation * w_sum[i_index] * outputs[i_index]
        return w_gradients

    def gradBDesc(self, bias, prevGrads, activations):
        b_gradients = np.empty(shape=bias.shape)
        w_sum = np.sum(prevGrads, axis=0)
        for i_index, i in enumerate(b_gradients):
                d_activation = activations[i_index] if activations[i_index] > 0 else activations[i_index] * self.leak
                i = d_activation * w_sum[i_index]
        return b_gradients

    def updateBias(self, bias, updates):
        for i_index, i in enumerate(updates):
                bias[i_index] += self.learningRate * i

    def updateWeights(self, weights, updates):
        for i_index, i in enumerate(updates):
            for j_index, j in enumerate(i):
                weights[i_index][j_index] += self.learningRate * j

    def gradDescSoftLog(self, weight, bias, gt, activations):
        w_gradients = np.empty(shape=weight.shape)
        b_gradients = np.empty(shape=bias.shape)
        for i_index, i in enumerate(w_gradients):
            for j_index, j in enumerate(i):
                j = (self.softmax[i_index] - gt[i_index]) * activations[j_index]
                b_gradients[i_index] = (self.softmax[i_index] - gt[i_index])
        return (w_gradients, b_gradients)

    def printShape(self):
        for weight in self.weights:
            print(weight.shape)
        for i in self.bias:
            print(i.shape)

shape = [12, 8, 6, 4, 2]
net = feedforward(shape)
print('prior weights', net.weights)
print('prior bias', net.bias)
x = np.random.rand(12)
y = np.array([1,0])
print('train data', x)
print('gt', y)
res = net.train(x)
loss = net.logLoss(y, res)
net.bProp(loss, y, res)
print('post weights', net.weights)
print('post bias', net.bias)

from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        return 0.5 * np.mean(np.sum(np.square(input - target), axis=1))

    def backward(self, input, target):
        return (input - target) / len(input)


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        e_input = np.exp(input)
        e_input_sum = np.repeat(np.sum(e_input, axis=1), 10, axis=0).reshape(e_input.shape)
        h_input = np.divide(e_input, e_input_sum)
        loss = -np.mean(np.sum(np.multiply(np.log(h_input), target), axis=1))
        # print(input[0])
        # print(e_input[0])
        # print(e_input_sum[0])
        # print(target[0])
        # print(loss)
        return loss

    def backward(self, input, target):
        '''Your codes here'''
        e_input = np.exp(input)
        e_input_sum = np.repeat(np.sum(e_input, axis=1), 10, axis=0).reshape(e_input.shape)
        h_input = np.divide(e_input, e_input_sum)
        return (h_input - target) / len(target)

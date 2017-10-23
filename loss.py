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
        e_input_sum = np.sum(e_input, axis=1)
        return -np.mean(np.sum(np.multiply(np.divide(e_input, e_input_sum), target), axis=1))

    def backward(self, input, target):
        '''Your codes here'''
        e_input = np.exp(input)
        e_input_sum = np.sum(e_input, axis=1)
        h_input = np.divide(e_input, e_input_sum)
        return (h_input - target) / len(target)

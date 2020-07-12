import numpy as np
import warnings
import functools
from abc import ABCMeta, abstractmethod
from neuron.activation_functions import Sigmoid, Function



def assert_shape(func):
    @functools.wraps(func)
    def wrapper(net_answers, y):
        assert net_answers.shape == y.shape, f'Neuron: Incorrect shapes: activations {net_answers.shape} != answers {y.shape}'
        return func(net_answers, y)
    return wrapper


class QuadraticCost(Function):
    
    @staticmethod
    @assert_shape
    def function(net_answers, y):
        return 0.5 * np.mean((net_answers - y) ** 2)

    @staticmethod
    @assert_shape
    def derivative(net_answers, y):
        return (net_answers - y) / y.shape[1]
    
    @staticmethod
    def delta(z, net_answers, y, activation_derivative):
        return QuadraticCost.derivative(net_answers, y) * activation_derivative(z)
    
class CrossEntropyCost(Function):
    
    @staticmethod
    @assert_shape
    def function(net_answers, y):
        return -np.mean(np.nan_to_num(y * np.log(net_answers) + (1 - y) * np.log(1 - net_answers)))
    
    @staticmethod
    @assert_shape
    def derivative(net_answers, y):
        return -np.nan_to_num(y / net_answers + (1 - y) / (1 - net_answers)) / y.shape[1]
   
    @staticmethod
    def delta(z, net_answers, y, activation_derivative):
        if activation_derivative == Sigmoid.function: # simplification
            return (net_answers - y)
        else:
            return (net_answers - y) / (net_answers * (1 - net_answers)) * activation_derivative(z)

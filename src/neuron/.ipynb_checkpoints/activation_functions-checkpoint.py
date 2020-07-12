import numpy as np
from abc import ABCMeta, abstractmethod

class Function():
    
    @abstractmethod
    def function(X, y):
        """
        net_answers - (n, K)
        y - correct answers on X (n, K)
        Return: scalar
        """
        pass
    
    @abstractmethod
    def derivative(net_answers, y):
        """
        Compute vector of partial derivatives (1st gear in chain rule)
        net_answers - prediction vector (K, B)
        y - vector of correct answers (K, B)
        Returns matrix (K, B)
        """
        pass
    
class Sigmoid(Function):
    
    def function(x):
        return 1 / (1 + np.exp(-x))

    def derivative(x):
        return Sigmoid.function(x) * (1 - Sigmoid.function(x))
    
class Linear(Function):

    def function(x):
        return x

    def derivative(x):
        return 1

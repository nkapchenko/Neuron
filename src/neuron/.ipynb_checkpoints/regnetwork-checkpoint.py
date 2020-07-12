from neuron.activation_functions import sigmoid, sigmoid_prime
from neuron.choice import J, J_derivative
from neuron.gradient import compute_grad_analytically
import random
import numpy as np
from matplotlib import pyplot as plt


class RegularizedNetwork(Network):

    def __init__(self, sizes, activation_function=sigmoid, activation_function_derivative=sigmoid_prime, l1=0, l2=0):
        super().__init__(sizes, activation_function, activation_function_derivative)
        np.random.seed(10)
        
        self.l1 = l1
        self.l2 = l2
        

    def update_batch(self, X, y, learning_rate, eps=0.001):
        """
        stochastic gradient descent using subset of data (выскочить из ямки)

        X - input matrix    (batch_size, m)
        y - correct answers (batch_size, K)

        Update neuron's weights with gradient.
        """
        assert  X.shape[1] == self.sizes[0],\
        f'Neuron: nb of features {X.shape[1]} != first layer size {self.sizes[0]}'
        assert  y.shape[1] == self.sizes[-1],\
        f'Neuron: input answers shape {y.shape[1]} != output layer shape {self.sizes[-1]}'

        dJdw, dJdb = self.backpropa(X, y)

        # Update weights
        self.weights = [w - learning_rate * djdw - self.l1 * np.sign(w) - self.l2 * w for w, djdw in zip(self.weights, dJdw)]
        self.biases =  [b - learning_rate * djdb for b, djdb in zip(self.biases,  dJdb)]


        

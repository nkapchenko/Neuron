from script.activation_functions import sigmoid, sigmoid_prime
from script.choice import J, J_derivative
from script.gradient import compute_grad_analytically
import random
import numpy as np


class Neuron:

    def __init__(self, weights, activation_function=sigmoid, activation_function_derivative=sigmoid_prime):
        """
        weights - of shape (m, 1), weights[0][0] - bias
        """

        assert weights.shape[1] == 1, "Incorrect weight shape"

        self.w = weights
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def neuron_answer(self, single_input):
        """
        One answer on one example.
        single_input of shape (m, 1) with first element = unity (for bias)
        Return: activation value
        """
        result = self.w.T.dot(single_input)
        return self.activation_function(result)

    # Examples x Features

    def summatory(self, input_matrix):
        """
        Summator function for all examples from input_matrix.
        input_matrix (n, m), each line - specific example,
        n - # examples, m - # parameters.
        self.w should be (m, 1)
        Return: vector (n, 1) with summator values for each example.
        """
        sumators = input_matrix.dot(self.w)
        return sumators

    def activation(self, summatory_activation):
        """Neuron activations for each example.
        summatory_activation - vector (n, 1),
        Return: vector (n, 1)
        """
        activations = self.activation_function(summatory_activation)
        return activations

    def neuron_answers(self, input_matrix):  # all answers to matrix X
        """
        Answers to all examples.
        input_matrix - matrix (n, m), each line - example
        n - # examples, m - # parameters.
        Return: float vector (n, 1)
        """
        return self.activation(self.summatory(input_matrix))

    def SGD(self, X, y, batch_size=1, learning_rate=0.1, eps=1e-6, max_steps=200):
        """
        Main gradient descent algorithm.
        X - input matrix (n, m)
        y - correct answers (n, 1)

        learning_rate - const
        batch_size - размер батча, на основании которого
        рассчитывается градиент и совершается один шаг алгоритма

        eps - 1st stopping criteria: diff in old vs new target function.
        Note: we could have also check the abs value of gradient.

        max_steps - 2nd stopping criteria: # of weight updates > max_steps

        Return: 1 if 1st stop (descent converged), 0 if 2nd stop (reached time limit)
        """

        i = 0
        while i < max_steps:
            i += 1

            Xy = np.append(X, y, axis=1)
            np.random.shuffle(Xy)
            mini_batches = np.array([Xy[k:k + batch_size] for k in range(0, len(Xy), batch_size)])

            for mini_batch in mini_batches:
                self.update_mini_batch( mini_batch, learning_rate, eps)

    def update_mini_batch(self, mini_batch, learning_rate, eps):
        """
        stochastic gradient descent using subset of data (выскочить из ямки
        mini_batch - matrix (batch_size, m + 1) with input matrix X and the last column y - answers
        """
        X, y = mini_batch[:, :-1], mini_batch[:, -1]
        y = y.reshape((len(y), 1))
        """
        X - input matrix (batch_size, m)
        y - correct answers (batch_size, 1)
        learning_rate - const
        eps - 1st stopping criteria: difference in old vs new target function.

        Compute gradient & update neuron's weights. If no target func perf return 1, else 0 (continue)
        """



        old_J = J(self, X, y)
        gradient = compute_grad_analytically(self, X, y, J_prime=J_derivative)
        self.w = self.w - learning_rate * gradient
        new_J = J(self, X, y)
        if abs(old_J - new_J) < eps:
            return 1
        return 0

    def __repr__(self):
        return f'{self.__class__.__name__}(' f'bias{self.w[0]} weights{self.w[1:]}'


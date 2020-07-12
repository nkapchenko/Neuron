from neuron.activation_functions import sigmoid, sigmoid_prime
from neuron.target_functions import J_quadratic, J_quadratic_derivative 
from neuron.gradient import compute_grad_analytically
import random
import numpy as np

J, J_derivative = J_quadratic, J_quadratic_derivative 

class Network:

    def __init__(self, sizes, activation_function=sigmoid, activation_function_derivative=sigmoid_prime):
        np.random.seed(10)
        self.nb_layers = len(sizes)
        self.sizes = sizes # ex. (748, 30 , 10)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]


        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        
        #debug
        keys = 'target_test  misclass_test target_valid misclass_valid'.split()
        self.debug = {key: [] for key in keys}

    def progonka(self, input_matrix):
        """ 
        input matrix (n, m)
        
        n - nb of examples
        m - nb of features
        K - nb of output neurons
        k_[i] - nb of neurons in layer i
        
        self.weights[i] - (k_[i+1], k_[i]); 
        self.biases[i] - (k_[i+1], 1)
        """
        assert input_matrix.shape[1] == 784, 'X should be (n, 784)'
        activation = input_matrix.T # (m, n)
        for b, w in zip(self.biases, self.weights):
            z = w.dot(activation) + b  # (k_[i], n)
            activation = self.activation_function(z)  # (k_[i], n)

        return activation.T # (n, K)




    def SGD(self, X, y, epochs, batch_size=1, learning_rate=1, eps=1e-6, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        X - (n, m)
        y - (n, K)

        eps - 1st stopping criteria: difference between old vs new target function (learning performance)
        Note: other choice would be to check the abs value of gradient.

        max_steps - 2nd stopping criteria: # of weight updates > max_steps

        Return: 1 if 1st stop (descent converged), 0 if 2nd stop (reached time limit)
        """
        assert isinstance(batch_size, int), 'batch_size should be int'
        
        if test_data:
            print(f'Initial random state: {self.evaluate(*test_data)}/{len(test_data[1])} classified. Target: {J(self.progonka(test_data[0]), test_data[1])}')
                    
        n = len(y)
        for _ in range(epochs):
            Xy = list(zip(X, y))
#             np.random.shuffle(Xy)
            batches = np.array([Xy[k:k + batch_size] for k in range(0, n, batch_size)])

            for batch in batches:
                Xbatch, ybatch= [np.asarray(x) for x in zip(*batch)]
                self.update_batch(Xbatch, ybatch, learning_rate, eps)
            
            valid_ok = self.evaluate(X, y)
            
            self.debug['target_test'].append(J(self.progonka(X), y))
            self.debug['misclass_test'].append(self.evaluate(X, y))
            
            if test_data:
                self.debug['target_valid'].append(J(self.progonka(test_data[0]), test_data[1]))
                self.debug['misclass_valid'].append(self.evaluate(*test_data))
            
            valid_ok = self.evaluate(*test_data)
            print(f'Epoch {_} done. Valid: {valid_ok}/{len(test_data[1])}.')

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
        self.weights = [w - learning_rate * djdw for w, djdw in zip(self.weights, dJdw)]
        self.biases =  [b - learning_rate * djdb for b, djdb in zip(self.biases,  dJdb)]



    def __repr__(self):
        return f'{self.__class__.__name__}(' \
            f'biases{self.biases} \n weights{self.weights}\n'


    def backpropa(self, X, answers):
        """ 
        X       - (B, m)
        answers - (B, K)
        
        B - batch size (intead of full examples n)
        m - nb of features
        K - nb of output neurons
        k_[i] - nb of neurons in layer i
        
        weights - [(k_[1], m) , (k_[2], k_[1]), ... , (K, k_[-1])]
        biases  - [(k_[1], 1) , (k_[2], 1) , ... , (K, 1)]
        acts    - [(k_[1], B) , (k_[2], B) , ... , (K, B)]
        zs      - [(k_[1], B) , (k_[2], B) , ... , (K, B)]
        deltas  - [(k_[1], B) , (k_[2], B) , ... , (K, B)]
        
        return 
        dJdw [(k_[1], m) , (k_[2], k_[1]), ... , (K, k_[-1])] (same sizes as weights)
        dJdb [(k_[1], 1) , (k_[2], 1) , ... , (K, 1)]  (same sizes as biases)
        """
        
        # preprocessing
        activation = X.T
        answers = answers.T # (K, B)
        activations = [activation]
        zs = []
        
        # Forward
        for w, b in zip(self.weights, self.biases): 
            z = w.dot(activation) + b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)
        

        # Backpropa
        network_answers = activations[-1] # (K, B) and answers (K, B)
        delta = J_derivative(network_answers, answers) * self.activation_function_derivative(zs[-1])
        deltas = [delta]
        for L in range (2, self.nb_layers):
            delta = self.weights[-L+1].T.dot(delta) * self.activation_function_derivative(zs[-L])
            deltas = [delta] + deltas

        # Gradients
        
        dJdb = [delta.sum(axis=1).reshape(len(delta),1) for delta in deltas]
        dJdw = [delta.dot(a.T) for delta, a in zip(deltas, activations[:-1])]

        return dJdw, dJdb
    
    def evaluate(self, Xtest, ytest):
        
        Xacts = self.progonka(Xtest)
        Xans = [np.argmax(act) for act in list(Xacts)]
        yans = [np.argmax(y) for y in ytest]     

        return sum(int(x == y) for (x, y) in zip(Xans, yans))
    
from neuron.activation_functions import Sigmoid, Linear
from neuron.target_functions import QuadraticCost, CrossEntropyCost
from neuron import vizualisation as vizual
import random
import numpy as np
from matplotlib import pyplot as plt


class Network:

    def __init__(self, sizes, activation=Sigmoid, cost=QuadraticCost, l1=0, l2=0):
#         np.random.seed(10)
        self.nb_layers = len(sizes) # ex. 3
        self.sizes = sizes # ex. (748, 30 , 10)
        self.biases = [np.random.randn(k, 1) for k in sizes[1:]]
        self.weights = [np.random.randn(L, k) / np.sqrt(k) for k, L in zip(sizes[:-1], sizes[1:])]


        self.activation = activation
        self.cost = cost
        self.l1 = l1
        self.l2 = l2
        
        #typical network representor
        self.x = np.random.randn(self.sizes[0], 1)
        
        #debug
        self.isdebug = False
        keys = 'djdw djdb deltas activations weights biases target_train misclass_train misclass_test target_test target_valid misclass_valid'.split()
        self.debug = {key: [] for key in keys}


    def predict(self, input_matrix):
        """ 
        Forward pass.
        input matrix (n, m)
        
        n - nb of examples
        m - nb of features
        K - nb of output neurons
        k_[i] - nb of neurons in layer i
        
        self.weights[i] - (k_[i+1], k_[i]); 
        self.biases[i] - (k_[i+1], 1)
        
        return (n,K) network output activations
        """
        activation = input_matrix.T # (m, n)
        for b, w in zip(self.biases, self.weights):
            z = w.dot(activation) + b  # (k_[i], n)
            activation = self.activation.function(z)  # (k_[i], n)

        return activation.T # (n, K)




    def fit(self, X, y, epochs=1, batch_size=1, learning_rate=1, eps=1e-6, test_data=None, valid_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        X - (n, m)
        y - (n, K)
        
        batch_size = number of elements in batch 
        eps - 1st stopping criteria: difference between old vs new target function (learning performance)
        test_data - tuple with (Xtest, ytest)
        Note: other choice would be to check the abs value of gradient.

        max_steps - 2nd stopping criteria: # of weight updates > max_steps

        Return: 1 if 1st stop (descent converged), 0 if 2nd stop (reached time limit)
        """
        assert isinstance(batch_size, int), 'batch_size should be int'
        n = len(y)
        
        if test_data:
            print(f'Training on {n} examples during {epochs} epochs. {batch_size} elements in batch \n \
                  Initial random state: {self.evaluate(*test_data)}% classified.\n \
                  Target: {self.cost.function(self.predict(test_data[0]), test_data[1])}')
          
        for _ in range(epochs):
            Xy = list(zip(X, y))
            np.random.shuffle(Xy)
            batches = np.array([Xy[k:k + batch_size] for k in range(0, n, batch_size)])

            for batch in batches:
                Xbatch, ybatch= [np.asarray(x) for x in zip(*batch)]
                self.update_batch(Xbatch, ybatch, learning_rate, eps)
            
            
            
            
            # Debug traces 
            self.debug['target_train'].append(self.cost.function(self.predict(X), y))
            self.debug['misclass_train'].append(self.evaluate(X, y))
            
            if epochs<=100:
                print(f'Epoch {_} ended') 
            if test_data:
                self.debug['target_test'].append(self.cost.function(self.predict(test_data[0]), test_data[1]))
                self.debug['misclass_test'].append(self.evaluate(*test_data))
                
                valid_ok = self.evaluate(*test_data)
                print(f'Test_data: {valid_ok}% correcly classified.')
                         
            if valid_data:
                self.debug['target_valid'].append(self.cost.function(self.predict(valid_data[0]), valid_data[1]))
                self.debug['misclass_valid'].append(self.evaluate(*valid_data))
                  
        print('')    
        print(f'Final network state: {self.evaluate(*test_data)}% classified. Target: {self.cost.function(self.predict(test_data[0]), test_data[1])}')
        
        

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
        self.weights = [w - self.l1 * np.sign(w) - self.l2 * w - learning_rate * djdw for w, djdw in zip(self.weights, dJdw)]
        self.biases =  [b - learning_rate * djdb for b, djdb in zip(self.biases,  dJdb)]



    def __repr__(self):
        return f'{self.__class__.__name__}(' \
            f'biases{self.biases} \n weights{self.weights}\n'


    def backpropa(self, X, answers):
        """ 
        X       - (B, m)
        answers - (B, K)
        
        B - batch size (nb of examples)   ex. 10
        m - nb of featuresm               ex. 784
        K - nb of output neurons          ex.10
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
            activation = self.activation.function(z)
            activations.append(activation)
        

        # Backpropa
        network_answers = activations[-1] # (K, B)
        
        # J_derivative * activation_derivative
        delta = self.cost.delta(zs[-1], network_answers, answers, activation_derivative=self.activation.derivative)

#         j_deriv = self.cost.derivative(network_answers, answers)
#         delta = j_deriv * self.activation.derivative(zs[-1]) # (K, B)
        
        
        deltas = [delta]
        for L in range (2, self.nb_layers):
            delta = self.weights[-L+1].T.dot(delta) * self.activation.derivative(zs[-L])
            deltas = [delta] + deltas

        # Gradients
        
        dJdb = [delta.sum(axis=1).reshape(len(delta),1) for delta in deltas]
        dJdw = [delta.dot(a.T) for delta, a in zip(deltas, activations[:-1])]
        
        # debug & vizualization
        if self.isdebug:
            self.debug['weights'].append(self.weights)
            self.debug['biases'].append(self.biases)
            self.debug['deltas'].append(deltas)
            self.debug['djdw'].append(dJdw)
            self.debug['djdb'].append(dJdb)
            self.debug['activations'].append(activations)

        return dJdw, dJdb
    
    
    def evaluate(self, X, y):
        
        Xacts = self.predict(X)
        Xans = [np.argmax(act) for act in list(Xacts)]
        yans = [np.argmax(y) for y in y]     

        return sum(int(x == y) for (x, y) in zip(Xans, yans))/len(yans)
    
    @property
    def vizualise(self):
        
        from neuron import vizualisation
        vizualisation.target(self.debug['target_train'], self.debug['target_test'], self.debug['target_valid'])
        vizualisation.misclassification(self.debug['misclass_train'], self.debug['misclass_test'], self.debug['misclass_valid'])
        
    def turn_debug(self, isdebug):
        self.isdebug = isdebug
        
        
    def train(self, *args):
        raise NameError('Neuron: please rename train function to fit')
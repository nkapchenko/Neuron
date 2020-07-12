"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple (training data, validation data, test data)

    ``training_data`` = numpy ndarray with 50,000 tuples (image, asnwer)
            >> image numpy ndarray with 784 values representing the 28 * 28 pixels
                    >> each value is pixel density in range [0, 1]
            >> answer digit values (0...9)

    ``validation_data`` and ``test_data`` are similar but with 10 000 images


    ``validation_data`` and ``test_data`` are similar, but only 10,000 images.

    For better format for ''training_data'' use ``load_data_wrapper()`` below
    """

    with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def perf_load_data_wrapper(nk=True):
    """ (c) Nikita
    X - (n, m)
    y - (n, K)
    """
    if nk:
        vectorized_result_func = nk_vectorized_result
    else:
        vectorized_result_func = vectorized_result

    tr_d, va_d, te_d = load_data()
    X, y = tr_d
    y = np.array([vectorized_result_func(v) for v in y])
    training_data = zip(X, y)
    
    validation_X, y2 = va_d
    validation_y =  np.array([vectorized_result_func(v) for v in y2])
    
    test_X, y3 = te_d
    test_y = np.array([vectorized_result_func(v) for v in y3])
    
    return (X, y), (validation_X, validation_y), (test_X, test_y)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def nk_vectorized_result(j):
    e = np.zeros(10)
    e[j] = 1.0
    return e
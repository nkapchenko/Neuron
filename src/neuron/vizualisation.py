import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")
sns.set(rc={'figure.figsize': (8, 6.)})
sns.set_style("whitegrid")

def target(training, test, valid):
    plt.figure()
    plt.title('Target function performance training vs validation data')
    plt.plot(training, label='training');
    plt.plot(test, label='test');
    plt.plot(valid, label='validation');
    plt.legend()
    
def misclassification(training, test, valid):
    plt.figure()
    plt.title('Classification accuracy')
    plt.plot(training, label='training');
    plt.plot(test, label='test');
    plt.plot(valid, label='validation');
    plt.legend()
    
    
def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    
    
def update_line(hl, new_data):
    hl.set_xdata(numpy.append(hl.get_xdata(), new_data))
    hl.set_ydata(numpy.append(hl.get_ydata(), new_data))
    plt.draw()
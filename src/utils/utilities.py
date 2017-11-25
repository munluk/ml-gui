import numpy as np
import matplotlib as mpl
import abc
# from abc import ABCMeta


def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))


# tanh function.
def tanh(x, deriv=False):
    if deriv:
        return 1-x*x
    return np.tanh(x)


# tanh function.
def linear(x, deriv=False):
    if deriv:
        return 1
    return x


def rectifier(x, deriv=False):
    if deriv:
        return x > 0.
    return x * (x > 0.)


def softmax(x, deriv=False, classify=False):
    """
     This is more numerically stable version of the softmax, than the naive implementation out of the book.
    :param x: scalar input
    :param deriv: Flag determines if the derivative should be returned
    :param classify: Flag determines if the output is used for classification
    :return:
    """
    if deriv:
        raise Exception('deriv is currently not implemented')
    if classify:
        return x == np.max(x, 0)[np.newaxis, :]
    max_vals = np.max(x, axis=0, keepdims=True)
    dif = x - max_vals
    return np.exp(dif - np.log(np.sum(np.exp(dif), axis=0, keepdims=True)))

'''
TODO: Softmax, RMSPROP, RPOP, plot_decision_region, basis functions
'''


def generate_test_set():
    meshx, meshy = np.mgrid[-10:10:1000j, -10:10:1000j]
    positions = np.vstack([meshx.ravel(), meshy.ravel()])
    test_data = np.vstack((positions, np.ones((1, positions.shape[1]))))
    return test_data, meshx, meshy


def draw_decision_boundaries(axes, meshx, meshy, predicted_labels, labels):
    cmap = mpl.colors.ListedColormap(np.array(['red', 'green', 'blue'])[np.any(labels, axis=0)])
    axes.contourf(meshx, meshy, predicted_labels.reshape(meshx.shape), cmap=cmap, alpha=0.3)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class DatasetException(Exception):
    pass


class Classifier:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, samples, lables):
        pass

    @abc.abstractmethod
    def predict(self, samples):
        pass

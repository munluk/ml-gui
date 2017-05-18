import numpy as np

'''
Utilities
'''
def returnList(lst):
    if not lst:
        return [False]
    return lst

def sigmoid(x,deriv=False, classify=False):
    x = np.clip(x, -500, 500)
    if deriv:
        return x*(1-x)
    if classify:
        return x >= 0.5
    return 1/(1+np.exp(-x))

def tanh(x,deriv=False, classify=False):
    if deriv:
        return 1-x*x
    if classify:
        return x >= 0
    return np.tanh(x)

def linear(x,deriv=False):
    if deriv:
        return 1
    return x

def rectifier(x, deriv=False):
    if deriv:
        return x > 0.
    return x * (x > 0.)

def softmax(x, deriv=False, classify=False):
    # exp(zj - m - log{sum_i{exp(zi - m)}})

    if deriv:
        # deriv is currently not used!
        return ''
    if classify:
        # return np.argmax(network.ff[-1], 0)
        return x == np.max(x, 0)[np.newaxis,:]
    # This is a safer computation of the softmax, than the naive implementation out of the book
    maxvals = np.max(x, axis=0, keepdims=True)
    dif = x - maxvals
    return np.exp(dif - np.log(np.sum(np.exp(dif), axis=0, keepdims=True)))

def get_function_dict():
    return {'softmax': softmax, 'sigmoid': sigmoid, 'rectifier': rectifier, 'linear': linear, 'tanh': tanh}

import numpy as np
import copy
import os
from src.utils.ConfigProvider import ConfigProvider as cfg_provider
from src.utils.mlp_utilities import *
np.seterr(all='raise')


class MLP:
    # def __init__(self, n_hidden_layers, n_hidden_neurons, n_input_neurons, n_output_neurons, batchsize_train, out_transfer, stdDev, hid_transfer = sigmoid, loss_function =''):
    def __init__(self):
        cfg_dict = cfg_provider.get_mlp_config()
        func_dict = get_function_dict()
        self.n_hidden_layers = int(cfg_dict['input_neurons'])
        self.n_hidden_neurons = int(cfg_dict['neurons'])
        self.n_input_neurons = int(cfg_dict['input_neurons'])
        self.n_output_neurons = int(cfg_dict['output_neurons'])
        self.batchsize_train = int(cfg_dict['batch_size'])
        self.stdDev = float(cfg_dict['std_dev'])
        self.out_transfer = func_dict[cfg_dict['out_transfer']]
        self.hid_transfer = func_dict[cfg_dict['hid_transfer']]

        print ''
        self.weights = []
        self.bias = []
        self.ff = []
        self.weights_gradient = []
        self.weights_gradient_old = []
        self.bias_gradient = []
        self.bias_gradient_old = []
        self.loss = []
        self.r_weights = []            #used for rmsprop
        self.r_bias = []
        self.accuracy = []
        self.iterator = 0
        # input layer
        self.ff.append(np.zeros((self.n_input_neurons, 1), dtype=np.float64))

        # create first hidden layer
        self.weights.append(self.stdDev * np.random.random((self.n_hidden_neurons, self.n_input_neurons)).astype(np.float64))
        self.bias.append(self.stdDev * np.random.random((self.n_hidden_neurons, 1)).astype(np.float64))
        self.weights_gradient.append(np.zeros(self.weights[0].shape,dtype=np.float64))
        self.weights_gradient_old.append(np.zeros(self.weights[0].shape,dtype=np.float64))
        self.bias_gradient.append(np.zeros(self.bias[0].shape, dtype=np.float64))
        self.bias_gradient_old.append(np.zeros(self.bias[0].shape, dtype=np.float64))
        self.ff.append(np.zeros((self.n_hidden_neurons,1), dtype=np.float64))
        self.r_weights.append(np.zeros(self.weights_gradient[0].shape, dtype=np.float64))
        # add consecutive layers
        for i in range(1, self.n_hidden_layers):
            self.weights.append(self.stdDev * np.random.random((self.n_hidden_neurons, self.n_hidden_neurons)).astype(np.float64))
            self.bias.append(self.stdDev * np.random.random((self.n_hidden_neurons, 1)).astype(np.float64))
            self.weights_gradient.append(np.zeros(self.weights[i].shape, dtype=np.float64))
            self.weights_gradient_old.append(np.zeros(self.weights[i].shape, dtype=np.float64))
            self.bias_gradient.append(np.zeros(self.bias[i].shape, dtype=np.float64))
            self.bias_gradient_old.append(np.zeros(self.bias[i].shape, dtype=np.float64))
            self.ff.append(np.zeros((self.n_hidden_neurons,1), dtype=np.float64))
        # add output layer
        self.weights.append(self.stdDev * np.random.random((self.n_output_neurons, self.n_hidden_neurons)).astype(np.float64))
        self.bias.append(self.stdDev * np.random.random((self.n_output_neurons, 1)).astype(np.float64))
        self.weights_gradient.append(np.zeros(self.weights[-1].shape, dtype=np.float64))
        self.weights_gradient_old.append(np.zeros(self.weights[-1].shape, dtype=np.float64))
        self.bias_gradient.append(np.zeros(self.bias[-1].shape, dtype=np.float64))
        self.bias_gradient_old.append(np.zeros(self.bias[-1].shape, dtype=np.float64))
        self.ff.append(np.zeros((self.n_output_neurons,1), dtype=np.float64))

        # this delta is needed for stacking the network. It is the delta of the input layer
        self.input_delta = []

        # for rmsprop
        self.r_weights = copy.deepcopy(self.weights_gradient)
        self.r_bias = copy.deepcopy(self.bias_gradient)

        # self.load_state(path='/home/lukas/weights/2x800xSigmoidxSoftmax/', number=20)
        self.learning_rate = 0.0
        self.learning_rate_set = False

    def forward(self, X):
        """
        Forward feed of the input through the network
        :param: input
        :return: return activation of each neuron
        """
        # input layer
        self.ff[0] = X
        # hidden layer
        for x in range(1, np.shape(self.ff)[0]-1):
            self.ff[x] = self.hid_transfer(self.weights[x-1].dot(self.ff[x-1]) + self.bias[x-1])
        # output layer
        self.ff[-1] = self.out_transfer(self.weights[-1].dot(self.ff[-2]) + self.bias[-1])

    def compute_loss(self, z):
        """
        :param z: target label. For dimensions or shape look at the function "test(..)"
        computes the loss and the delta of the output layer
        """
        # softmax computation
        if self.out_transfer == softmax:
            tmp = self.ff[-1] * z
            self.loss.append(-np.sum(np.log(np.sum(tmp,0))))
            residual = z - self.ff[-1]
            delta = - residual
            return residual, delta

        residual = z - self.ff[-1]
        # print residual
        delta = - residual * self.out_transfer(self.ff[-1], deriv=True)
        self.loss.append(np.sum(residual ** 2, 1))
        return residual, delta

    def backprop(self, z):
        """
        :param z: labels of the current batch
        :return:
        """
        residual, delta = self.compute_loss(z)
        for i in range(np.shape(self.ff)[0] - 2, -1, -1):
            self.weights_gradient[i] = delta.dot(self.ff[i].T)/self.batchsize_train
            self.bias_gradient[i] = np.sum(delta, 1, keepdims=True)/self.batchsize_train
            delta = self.weights[i].T.dot(delta) * self.hid_transfer(self.ff[i], deriv=True)
        self.input_delta = delta

    def update_weights(self, learning_rate = 0.01, momentum_flag = True):
        """
        :param learning_rate: Learning rate for SGD
        :param momentum_flag: 'True' -> Uses momentum, 'False' -> standard SGD
        :return: []
        """
        momentum = 0.0
        if momentum_flag:
            momentum = 1.0 - learning_rate
        for i in range(0, np.shape(self.weights)[0]):
            self.weights[i] -= self.learning_rate * (self.weights_gradient[i] + momentum * self.weights_gradient_old[i])
            self.bias[i] -= self.learning_rate * (self.bias_gradient[i] + momentum * self.bias_gradient_old[i])
            # update the old gradient
            self.weights_gradient_old[i] = self.weights_gradient[i]
            self.bias_gradient_old[i] = self.bias_gradient_old[i]
        self.iterator += 1

    def update_weights_rmsprop(self, learning_rate=0.1, decay=0.9):
        """
        Implemtation of RMSPROP after "http://climin.readthedocs.io/en/latest/rmsprop.html#id1"
        :param learning_rate: Learning rate RMSPROP
        :param decay: Decay for RMSPROP
        :return: []
        """
        for i in range(0, np.shape(self.weights)[0]):
            # compute step width
            self.r_weights[i] = (1.0 - decay) * self.weights_gradient[i]**2 + decay * self.r_weights[i]
            self.r_bias[i] = (1.0 - decay) * self.bias_gradient[i]**2 + decay * self.r_bias[i]
            try:
                v_weights = learning_rate * self.weights_gradient[i] / np.sqrt(self.r_weights[i])
                v_bias = learning_rate * self.bias_gradient[i] / np.sqrt(self.r_bias[i])
            except Exception as e:
                self.r_weights[i] = np.zeros(self.r_weights[i].shape)
                self.r_bias[i] = np.zeros(self.r_bias[i].shape)
                if i == range(0, np.shape(self.weights)[0])[-1]:
                    self.update_weights(learning_rate=0.001, momentum_flag=True)
                print e.message
                print '[Backprop] Numerical problems. Changed from RMSPROP to SGD with Momentum'
                return
            # update weights
            self.weights[i] -= v_weights
            self.bias[i] -= v_bias
        self.iterator += 1

    def check_dimensions(self, X, z):
        """
        :param X: Input batch with shape (length of input-vector, batchsize_train)
        :param z: label batch (length of output-vector, batchsize_train)
        :return: []
        """
        alright = True
        # check dimensions at the first training iteration
        if np.shape(X)[1] != self.batchsize_train:
            alright = False
            raise Exception('[Network]: Batch size does not match dimension 1 in batch')
        if np.shape(z)[1] != self.batchsize_train:
            alright = False
            raise Exception('[Network]: Batch size does not match dimension 1 of labels in batch')
        if np.shape(X)[0] != self.n_input_neurons:
            alright = False
            raise Exception('[Network]: Lenght of input vector does not match the number of input neurons')
        if np.shape(z)[0] != self.n_output_neurons:
            alright = False
            raise Exception('[Network]: Length of label vector does not match the number of output neurons')
        if alright:
            print '[Network]: No problem with input and output dimensions!'

    def train(self, X, z):
        """
        :param X:               Input batch with shape (length of input-vector, batchsize)
        :param z:               label batch (length of output-vector, batchsize)
        :param learning_rate:   learning rate of gradient methods
        :param momentum_flag:   if 'True' training with momentum else not
        :param print_epoch:     if '1' prints every batch if '300' prints every 300-th batch and so forth
        :return:                []
        """
        X = X[:, 0:2].T
        z = z.T
        cfg_dict = cfg_provider.get_mlp_config()
        self.learning_rate = float(cfg_dict['learning_rate'])
        momentum_flag = bool(cfg_dict['momentum'])
        train_steps = int(cfg_dict['train_steps'])
        optimizer = cfg_dict['optimizer']
        print_batch=100
        for ii in xrange(0, train_steps):
            self.forward(X)
            self.backprop(z)
            # chose a learning procedure
            if optimizer == 'gradientDescent':
                self.update_weights(learning_rate=self.learning_rate, momentum_flag=momentum_flag)
            if optimizer == 'rmsprop':
                self.update_weights_rmsprop(learning_rate=self.learning_rate,decay=0.9)
            if np.mod(self.iterator + 1, print_batch) == False:
                print '[batch: %i] Loss: ' % (self.iterator + 1) + str(self.loss[-1])

    def test(self, X, z, classify):
        """
        :param X: Data of test batch. X must have shape (number of input dimensions, number of samples)
        :param z: Labels of test batch. z must have shape (number of output dimensions, number of samples)
        :param classify: This variable is ONLY necessary for sigmoid or tanh output functions!!!!! See transfer functions in the top of this file
        :return: Accuracy in terms of correctly classified samples given the test batch. This should ONLY BE USED FOR CLASSIFICATION
        """
        batchsize = np.shape(X)[1]
        self.forward(X)
        if self.out_transfer == softmax:
            acc = float(sum(np.argmax(z,0) == np.argmax(self.ff[-1],0)))/batchsize
        if (self.out_transfer == sigmoid) or (self.out_transfer == tanh):
            acc = np.float(np.sum((self.ff[-1] > 0.5) == np.array(z, dtype=np.bool)))/batchsize
        self.accuracy.append(acc)
        print '[Network] Accuracy on test-batch %f%%' % (acc * 100)
        return acc

    def predict(self, samples):
        X = samples[0:2, :]
        self.forward(X)
        if self.out_transfer == softmax:
            y = np.argmax(self.ff[-1], 0)
        if (self.out_transfer == sigmoid) or (self.out_transfer == tanh):
            y = self.ff[-1] > 0.5
        return y

    def save_state(self, path='/home/lukas/weights/'):
        """
        This function saves the weights and biases of the current network, as well as the parameters (number of hidden layers, neurons and so forth)
        :path: Path where to save the weights
        :return: []
        """
        stuff_in_path = os.listdir(path)
        counter = 0
        for i in stuff_in_path:
            if 'parameters' in i:
                counter += 1
        with open(path + 'info.txt', mode='a') as f:
            f.write('counter: %i \taccuracy: %.8f%% \tloss: %.8f\n' % (counter, returnList(self.accuracy)[-1] * 100, returnList(self.loss)[-1]))

        parameters = [  self.batchsize_train,
                        self.iterator,
                        self.n_hidden_layers,
                        self.n_hidden_neurons,
                        self.n_input_neurons,
                        self.n_output_neurons,
                        self.hid_transfer.__name__,
                        self.out_transfer.__name__]
        try:
            print '[Network] Saving network status ...'
            np.save(path + 'parameters' + str(counter), parameters)
            np.save(path + 'weights' + str(counter), self.weights)
            np.save(path + 'bias' + str(counter), self.bias)
            np.save(path + 'weights_gradient' + str(counter), self.weights_gradient)
            np.save(path + 'bias_gradient' + str(counter), self.bias_gradient)
            np.save(path + 'loss' + str(counter), self.loss)
            np.save(path + 'accuracy' + str(counter), self.accuracy)
            np.save(path + 'r_weights' + str(counter), self.r_weights)
            np.save(path + 'r_bias' + str(counter), self.r_bias)
            print '\033[92m' + '[Network] Network status succesfully saved' + '\033[0m'

        except Exception as e:
            print '\033[1m' + '\033[91m' + '[Network] Could not correctly save network status:' + '\033[0m'
            print e.message

    def load_state(self, path, number):
        self.weights = np.load(path + 'weights' + str(number) + '.npy')
        self.bias = np.load(path + 'bias' + str(number) + '.npy')
        self.weights_gradient = np.load(path + 'weights_gradient' + str(number) + '.npy')
        self.bias_gradient = np.load(path + 'bias_gradient' + str(number) + '.npy')
        self.loss = list(np.load(path + 'loss' + str(number) + '.npy')[:])
        self.accuracy = list(np.load(path + 'accuracy' + str(number) + '.npy')[:])

if __name__ == "__main__":

    samples = np.load('../datasets/samples.npy').T[0:2, :]
    labels = np.load('../datasets/labels.npy')[:, 0:2].T
    print 'datasets loaded'

    batchsize = samples.shape[1]
    network = MLP()
    X = samples
    z = labels
    network.train(X, z)

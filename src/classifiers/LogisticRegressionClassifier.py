import numpy as np
import src.utils.utilities as ut


class LogisticRegressionClassifier:
    def __init__(self):
        """
        Initialization of the logistic classifier.
        :param x_train: training samples must be of shape (dimension of one sample, number of samples)
        :param z_train: training labels must be of shape (number of samples, number of classes) -> USE ONE-HOT ENCDOING
        """
        # check if the data set contains two classes only
        # if sum(np.any(z_train, axis=0)) > 2:
        #     raise Exception('Too many Classes. Binary Logistic regression only works only on a two-class problem')
        self.x_train = np.array([], dtype=np.float64)
        self.z_train = np.array([], dtype=np.float64)
        self.weights = 0.1 * np.random.randn(3, 1)
        self.loss = []
        self.gradient = np.zeros((3, 1))
        self.learn_rate = 0.1

    def train(self, x_train, z_train):
        self.init_from_data(x_train, z_train)
        for i in range(0, 10000):
            self.compute_loss()
            self.compute_gradients()
            self.update_weights()
            # if np.mod(i+1, 100) == 0:
                # print self.loss[-1]

    def predict(self, x):
        y = ut.sigmoid(self.weights.T.dot(x))
        return np.array(y >= 0.5).astype(int)

    def compute_loss(self):
        # y = self.predict(self.x_train)
        y = ut.sigmoid(self.weights.T.dot(self.x_train))
        self.loss.append(-np.sum(self.z_train * np.log(y) + (1 - self.z_train) * np.log(1 - y)))

    def compute_gradients(self):
        no_samples = self.z_train.shape[1]
        self.gradient = np.sum((ut.sigmoid(self.weights.T.dot(self.x_train)) - self.z_train) * self.x_train,
                               axis=1,
                               keepdims=True) / no_samples

    def update_weights(self):
        self.weights -= self.learn_rate * self.gradient

    def init_from_data(self, samples, labels):
        # check if the data set contains two classes only
        if sum(np.any(labels, axis=0)) > 2:
            raise Exception('Too many Classes. Binary Logistic regression only works only on a two-class problem')
        self.x_train = samples.T
        # Chose the labeling such that class 0 matches the first color that is contained in the data set. The color
        # sequence is red, green, blue and this is also the way the samples are stacked together. If the data set
        # contains only [red, blue], then  class 0 must belong to red. If data set contains only [green, blue], then
        # class 0 must belong to green and so forth
        self.z_train = (labels[:, np.any(labels, axis=0)][:, 1:2]).T

if __name__ == '__main__':
    samples = np.load('../datasets/samples.npy')
    labels = np.load('../datasets/labels.npy')
    log_reg = LogisticRegressionClassifier()
    log_reg.train(samples, labels)

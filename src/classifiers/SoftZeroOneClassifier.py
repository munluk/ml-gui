import numpy as np
import src.utils.utilities as ut


class SoftZeroOneClassifier:
    def __init__(self):
        """
        Initialization of the logistic classifier.
        :param x_train: training samples must be of shape (dimension of one sample, number of samples)
        :param z_train: training labels must be of shape (number of samples, number of classes) -> USE ONE-HOT ENCDOING
        """
        self.weights = 2 * np.random.randn(3, 1)
        self.x_train = np.array([], dtype=np.float64)
        self.z_train = np.array([], dtype=np.float64)
        self.loss = []
        self.gradient = np.zeros((3, 1))
        self.prediction_train = 0.0

        # parameters
        self.learn_rate = 0.01
        self.beta = 1.0
        self.lambda_regularize = 2.0

    def train(self, x_train, z_train):
        self.init_from_data(x_train, z_train)
        for i in range(0, 10000):
            self.compute_loss()
            self.compute_gradients()
            self.update_weights()
            if not np.mod(i+1, 100):
                print self.loss[-1]

    def predict(self, x):
        y = ut.sigmoid(self.weights.T.dot(x))
        return np.array((y >= 0.5)).astype(int)

    def compute_loss(self):
        y = ut.sigmoid(self.beta * (self.weights.T.dot(self.x_train)))
        self.prediction_train = y
        self.loss.append(-np.sum((y - self.z_train)**2) + self.lambda_regularize*self.weights.T.dot(self.weights))

    def compute_gradients(self):
        y = self.prediction_train
        y2 = y**2
        y3 = y2*y
        self.gradient = np.sum((-y3 + y2 + y2 * self.z_train - y * self.z_train) * self.beta * self.x_train,
                               axis=1, keepdims=True) + self.lambda_regularize * self.weights

    def update_weights(self):
        self.weights -= self.learn_rate * self.gradient

    def init_from_data(self, samples, labels):
        # check if the data set contains two classes only
        if sum(np.any(labels, axis=0)) > 2:
            raise Exception('Too many Classes. Binary SoftZeroOne Classifier only works only on a two-class problem')
        self.x_train = samples.T
        # Chose the labeling such that class 0 matches the first color that is contained in the data set. The color
        # sequence is red, green, blue and this is also the way the samples are stacked together. If the data set
        # contains only [red, blue], then  class 0 must belong to red. If data set contains only [green, blue], then
        # class 0 must belong to green and so forth
        self.z_train = (labels[:, np.any(labels, axis=0)][:, 1:2]).T.astype(np.float64)

if __name__ == '__main__':
    samples = np.load('../datasets/samples.npy')
    labels = np.load('../datasets/labels.npy')
    log_reg = SoftZeroOneClassifier()
    log_reg.train(samples, labels)

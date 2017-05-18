import numpy as np
import src.utils.utilities as ut


class SoftMaxRegressionClassifier:
    def __init__(self):
        """
        Initialization of the logistic classifier.
        :param x_train: training samples must be of shape (dimension of one sample, number of samples)
        :param z_train: training labels must be of shape (number of samples, number of classes) -> USE ONE-HOT ENCDOING
        """
        self.weights = np.array([], dtype=np.float64)
        self.x_train = np.array([], dtype=np.float64)
        self.z_train = np.array([], dtype=np.float64)
        self.output = np.array([], dtype=np.float64)
        self.loss = []
        self.gradient = np.zeros(self.weights.shape)
        self.learn_rate = 0.1

    def train(self, samples, labels):
        self.init_from_data(samples, labels)
        for i in range(0, 10000):
            self.compute_loss()
            self.compute_gradients()
            self.update_weights()
            if np.mod(i+1, 100) == 0:
                print self.loss[-1]

    def predict(self, x):
        y = ut.softmax(self.weights.dot(x))
        return np.argmax(y, axis=0)

    def compute_loss(self):
        self.output = ut.softmax(self.weights.dot(self.x_train)) * self.z_train
        self.loss.append(-np.sum(np.log(np.sum(self.output, axis=0))))

    def compute_gradients(self):
        no_samples = self.z_train.shape[1]
        residual = self.output - self.z_train
        self.gradient = residual.dot(self.x_train.T) / no_samples

    def update_weights(self):
        self.weights -= self.learn_rate * self.gradient

    def init_from_data(self, samples, labels):
        n_classes = np.sum(np.any(labels, axis=0))
        self.weights = 2.0 * np.random.randn(n_classes, 3) + np.random.randn(n_classes, 3) * 5
        self.x_train = samples.T
        # Chose the labeling such that class 0 matches the first color that is contained in the data set. The color
        # sequence is red, green, blue and this is also the way the samples are stacked together. If the data set
        # contains only [red, blue], then  class 0 must belong to red. If data set contains only [green, blue], then
        # class 0 must belong to green and so forth
        self.z_train = labels[:, np.any(labels, axis=0)].T



if __name__ == '__main__':
    samples = np.load('../datasets/samples.npy')
    labels = np.load('../datasets/labels.npy')
    log_reg = SoftMaxRegressionClassifier()
    log_reg.train(samples, labels)

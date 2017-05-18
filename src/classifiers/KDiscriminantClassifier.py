import numpy as np


class KDiscriminantClassifier:
    def __init__(self):
        self.x_train = np.array([], dtype=np.float64)
        self.z_train = np.array([], dtype=np.float64)
        self.weights = np.zeros((3, 3))

    def train(self, samples, labels):
        self.x_train = samples
        self.z_train = labels
        self.weights = np.linalg.inv(self.x_train.T.dot(self.x_train)).dot(self.x_train.T).dot(self.z_train)

    def predict(self, x):
        return np.argmax(self.weights.T.dot(x), axis=0)


if __name__ == '__main__':
    samples = np.load('../datasets/samples.npy')
    labels = np.load('../datasets/labels.npy')
    log_reg = KDiscriminantClassifier()
    log_reg.train(samples, labels)
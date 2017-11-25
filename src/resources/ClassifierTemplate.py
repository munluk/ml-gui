# This script is a template for a discriminant classifier to work with the DiscriminantClassifierGUI.np.
# If you are creating a binary classifier, then check out the training labels as shown in the __init__ method.
#
# IMPORTANT: 1. Script name must be identical to the class name of the classifier you are implementing.
#            2. You can modify this script as you want, but the header of the methods __init__, train and predict may
#               not be changed. Otherwise they won't work in the tool.


import numpy as np
from src.utils.utilities import Classifier


class TemplateClassifier(Classifier):
    def __init__(self, x_train, z_train):
        """
        Initialization of the your classifier.
        :param x_train: training samples must be of shape < insert Shape for your classifier >
        :param z_train: training labels must be of shape < insert Shape for your classifier >
        """
        # check if the data set contains two classes only
        if sum(np.any(z_train, axis=0)) > 2:
            raise Exception(
                'Too many Classes. Binary <insert name of your Classifier> only works only on a two-class problem')

        self.x_train = x_train.T
        self.z_train = z_train

        self.loss = []
        self.weights = np.array([])
        self.gradient = np.zeros(self.weights.shape)

    def train(self, x_train, z_train):
        print 'train'
        # self.predict(self.x_train)
        self.compute_loss()
        self.compute_gradients()
        self.update_weights()

    def predict(self, x):
        print 'predict'
        return

    def compute_loss(self):
        print 'compute loss'
        self.loss.append([])

    def compute_gradients(self):
        print 'compute gradients'

    def update_weights(self):
        print 'update weights'


'''
Testing the script. The section below the if statement is executed if you start run script as main and not from any
other function/class/script. If you want to test the classifier alone, then you need to load a data set. You can
generate it with GUI and then save it by pressing the 'save' button.
'''
if __name__ == '__main__':
    samples = np.load('../datasets/samples.npy')
    labels = np.load('../datasets/labels.npy')
    log_reg = TemplateClassifier(samples, labels)
    log_reg.train()
